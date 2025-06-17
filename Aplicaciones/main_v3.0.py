import pandas as pd
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings, Document, StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.llms.openai_like import OpenAILike # O el LLM que uses
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter # Para filtros
from llama_index.readers.file import PDFReader

import requests
import time
import asyncio 
import pypandoc
from typing import Any, List
import os
import datetime

FICHERO_INDICE = "../Datos/indice.xlsx"
CARPETA_DATOS_SALVADOS = "../Datos/data_storage/"
CARPETA_ENTRADA_DOCUMENTOS = "../Nuevos_Documentos/" # Carpeta donde se encuentran los documentos a procesar
CARPETA_DOCUMENTOS_PROCESADOS = "../Documentos/"
NUMERO_MINIMO_PALABRAS = 0 # Número mínimo de palabras para considerar un trozo de texto
UMBRAL_MAXIMO_PALABRAS = 300 # Número máximo de palabras para considerar un trozo de texto
PALABRAS_SOLAPAMIENTO = 50 # Número de palabras que se solapan entre trozos consecutivos
MODELO_EMBEDDINGS = "text-embedding-nomic-embed-text-v1.5-embedding"
MODELO_LLM = "google/gemma-3-12b" # Nombre del modelo LLM a usar
SERVIDOR_LMSTUDIO = "http://192.168.20.10:1234/v1" # URL del servidor LM Studio

class LMStudioEmbedding(BaseEmbedding):
    """
    Clase de Embedding personalizada para interactuar con un endpoint tipo OpenAI
    servido localmente (ej: LM Studio) sin validación estricta del nombre del modelo.
    """
    _model_name: str = PrivateAttr()
    _api_base: str = PrivateAttr()
    _api_key: str = PrivateAttr()
    _session: requests.Session = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://127.0.0.1:8080/v1",
        api_key: str = "lm-studio",
        **kwargs: Any,
    ) -> None:
        # Llama al __init__ de la clase base (BaseEmbedding)
        super().__init__(**kwargs)
        self._model_name = model_name
        self._api_base = api_base.strip('/')
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            # "Authorization": f"Bearer {self._api_key}" # Si se necesita
        })

    # --- Métodos Síncronos (Implementados) ---
    def _get_query_embedding(self, query: str) -> List[float]:
        """Obtiene el embedding para una consulta (query)."""
        return self._get_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Obtiene el embedding para un fragmento de texto (documento)."""
        return self._get_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Obtiene embeddings para una lista de textos (batch)."""
        results = []
        for text in texts:
             # Reutiliza _get_embedding para cada texto
            embedding = self._get_embedding(text)
            if embedding: # Asegúrate de que no sea None/vacío si _get_embedding puede devolver eso
                 results.append(embedding)
            # Pequeña pausa
            time.sleep(0.05) # Considera hacerlo configurable o eliminarlo si no es necesario
        return results

    async def _aget_query_embedding(self, query: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)

    def _get_embedding(self, text: str) -> List[float]:
        """Método central para llamar a la API de embedding."""
        if not text:
             print("Warning: Intentando obtener embedding para texto vacío.")
             return []

        payload = {
            "model": self._model_name,
            "input": text.strip() # Quitar espacios extra al inicio/final
        }
        embedding_url = f"{self._api_base}/embeddings"

        try:
            response = self._session.post(embedding_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                if "embedding" in data["data"][0] and isinstance(data["data"][0]["embedding"], list):
                    return data["data"][0]["embedding"]
                else:
                     raise ValueError(f"API response missing 'embedding' in data[0]. Response: {data}")
            else:
                 raise ValueError(f"API response missing 'data' or 'data' is empty. Response: {data}")

        except requests.exceptions.RequestException as e:
            print(f"Error calling embedding API at {embedding_url}: {e}")
            raise
        except Exception as e:
            print(f"Error processing embedding API response: {e}")
            raise

def load_llm():
    """
    Carga todo el entorno de LLM Studio y el modelo LLM
    """
    global llm, semantic_splitter

    # 1. Instancia tu embedding personalizado (Ahora debería funcionar)
    custom_embed_model = LMStudioEmbedding(
        model_name=MODELO_EMBEDDINGS,
        api_base=SERVIDOR_LMSTUDIO
    )

    # 2. Configura LlamaIndex globalmente
    Settings.embed_model = custom_embed_model

    # 3. Configura tu LLM 
    llm = OpenAILike(
        model=MODELO_LLM, 
        api_base=SERVIDOR_LMSTUDIO,
        api_key="lm-studio",
        is_chat_model=True,
    )
    Settings.llm = llm

    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1, # Cuántas oraciones "mirar hacia atrás y adelante" para suavizar
        breakpoint_percentile_threshold=95, # Umbral para decidir un punto de división
        embed_model=Settings.embed_model # Usa el embed_model global configurado
    )

def leer_archivos_carpeta(ruta_carpeta):
  """
  Analiza los archivos dentro de una carpeta y devuelve información sobre ellos.

  Args:
    ruta_carpeta (str): La ruta a la carpeta que contiene los archivos.

  Returns:
    list: Una lista de diccionarios. Cada diccionario representa un archivo
          y contiene las claves: 'carpeta', 'nombre', 'extension', 'ruta_completa'.
          Devuelve una lista vacía si la carpeta no existe o está vacía.
          Imprime errores si ocurre un problema al acceder a la carpeta.
  """
  lista_info_archivos = []
  try:
    # 1. Listar todos los elementos (archivos y carpetas) en la ruta especificada
    nombres_elementos = os.listdir(ruta_carpeta)
  except FileNotFoundError:
    print(f"Error: La carpeta '{ruta_carpeta}' no fue encontrada.")
    return [] # Devuelve lista vacía si la carpeta no existe
  except Exception as e:
    print(f"Error al acceder a la carpeta '{ruta_carpeta}': {e}")
    return []

  # 2. Iterar sobre cada nombre de archivo/carpeta encontrado
  for nombre_elemento in nombres_elementos:
    # 3. Construir la ruta completa al elemento (puede ser archivo o carpeta)
    # Usamos abspath para asegurarnos de tener la ruta completa y absoluta
    ruta_completa_elemento = os.path.abspath(os.path.join(ruta_carpeta, nombre_elemento))

    # 4. Verificar si la ruta completa es realmente un archivo (y no una subcarpeta)
    if os.path.isfile(ruta_completa_elemento):
      try:
        # 5. Extraer la información requerida usando os.path
        carpeta_contenedor = os.path.dirname(ruta_completa_elemento)
        nombre_base, extension_con_punto = os.path.splitext(nombre_elemento) # Usamos nombre_elemento aquí para obtener solo el nombre del archivo
        # Quitamos el punto inicial de la extensión, si existe
        extension = extension_con_punto[1:] if extension_con_punto else ""

        # 6. Crear el diccionario con la información
        info_archivo = {
            'carpeta': carpeta_contenedor,
            'nombre': nombre_base,
            'extension': extension,
            'nombre_con_extension': nombre_elemento,
            'ruta_completa': ruta_completa_elemento
        }

        # 7. Agregar el diccionario a la lista
        lista_info_archivos.append(info_archivo)
      except Exception as e:
         # Captura errores inesperados al procesar un archivo específico
         print(f"Error procesando el elemento '{nombre_elemento}': {e}")
  print(f"Se leyeron {len(lista_info_archivos)} documentos de la carpeta {ruta_carpeta}.")
  return lista_info_archivos # Devuelve la lista de archivos procesados    

def load_index():
    """
    Carga el índice de trozos de documentos y metadatos que ha sido salvado previamente
    """
    global index

    if not os.path.exists(CARPETA_DATOS_SALVADOS + "index_store.json"):
        index = None
        print("No existe ningún índice salvado.")
    else:
        storage_context = StorageContext.from_defaults(persist_dir=CARPETA_DATOS_SALVADOS)
        index = load_index_from_storage(storage_context)
        print("Índice cargado con éxito.")
    return index

def split_text_with_overlap(text, chunk_size_words, overlap_words):
    """
    Divide un texto en trozos más pequeños basados en número de palabras,
    con un solapamiento especificado (en palabras).

    Args:
        text (str): El texto a dividir.
        chunk_size_words (int): El número máximo de palabras deseado por trozo.
        overlap_words (int): El número de palabras que se solaparán
                               entre trozos consecutivos.

     Returns:
         list[str]: Una lista con los trozos de texto (sub-chunks).
    """
    if chunk_size_words <= overlap_words:
         raise ValueError("El tamaño del trozo (chunk_size_words) debe ser mayor que el solapamiento (overlap_words).")

    words = text.split()
    if not words:
         return [] # Devolver lista vacía si no hay texto

    sub_chunks = []
    
    start_idx = 0
    # El paso avanza para asegurar el solapamiento en la siguiente iteración
    step = chunk_size_words - overlap_words 
    
    while start_idx < len(words):
        end_idx = start_idx + chunk_size_words
        word_slice = words[start_idx:end_idx] # La rebanada de palabras
        sub_chunks.append(" ".join(word_slice))
        
        # Avanzar al siguiente punto de inicio
        start_idx += step
            
    return sub_chunks

def digitaliza_un_documento(documento, fila_indice):
    """
    Procesa un documento y lo digitaliza.
    """
    # Aquí iría la lógica para digitalizar el documento
    print(f"Digitalizando <<<{documento['nombre_con_extension']}>>>")

    resultado = []

    try:
        if documento['extension'] == 'pdf':
            pdf_reader = PDFReader(return_full_document=True)
            
            reader = SimpleDirectoryReader(input_files=[documento['ruta_completa']],file_extractor={".pdf": pdf_reader})
            documents = reader.load_data()

            #    Extrae el texto de cada objeto 'Document' y se unen los metadatos externos con los propios del documento.
            if documents:
                # Asumimos que solo hay un documento porque return_full_document=True
                full_pdf_document = documents[0]

                # Transforma el documento completo en una lista de nodos (chunks semánticos)
                # Esto reemplaza tu lógica de split_text_with_overlap
                semantic_nodes = semantic_splitter.get_nodes_from_documents([full_pdf_document])
                
                indice_en_documento = 0
                for node in semantic_nodes:
                    # Los nodos (chunks) ya vienen con sus propios metadatos, puedes acceder a ellos
                    # como node.text para el contenido y node.metadata para los metadatos generados por el parser.
                    
                    if len(node.text.split()) >= NUMERO_MINIMO_PALABRAS:
                        nuevo_texto = {'texto': node.text, 'metadatos': fila_indice.to_dict()}
                        
                        # Agrega los metadatos originales del documento (si los hay en full_pdf_document.metadata)
                        # y los metadatos generados por el NodeParser (si los quieres preservar).
                        # Node.metadata ya incluye el 'file_name', 'page_label' (si aplica), etc.
                        nuevo_texto['metadatos'].update(full_pdf_document.metadata) 
                        nuevo_texto['metadatos'].update(node.metadata) # Incluye metadatos del nodo (ej. sección, etc.)

                        nuevo_texto['metadatos']['LongitudTexto'] = len(node.text)
                        nuevo_texto['metadatos']['NumeroPalabras'] = len(node.text.split())
                        nuevo_texto['metadatos']['NumeroTrozoEnDocumento'] = indice_en_documento
                        nuevo_texto['metadatos']['IdTrozoTexto'] = fila_indice['Identificador'] + '-' + str(indice_en_documento)
                        
                        resultado.append(nuevo_texto)
                        indice_en_documento += 1
            else:
                print("No se pudo extraer texto del documento o el documento está vacío.")
        # if documento['extension'] == 'pdf':
        #     pdf_reader = PDFReader(return_full_document=True)
        #     reader = SimpleDirectoryReader(input_files=[documento['ruta_completa']],file_extractor={".pdf": pdf_reader})
        #     documents = reader.load_data()

        #     #    Extrae el texto de cada objeto 'Document' y se unen los metadatos externos con los propios del documento.
        #     if documents:
        #         indice_en_documento = 0
        #         for trozo_texto in documents:
        #             if len(trozo_texto.text.split()) >= NUMERO_MINIMO_PALABRAS:
        #                 nuevos_trozos_texto = split_text_with_overlap(trozo_texto.text, UMBRAL_MAXIMO_PALABRAS, PALABRAS_SOLAPAMIENTO)
        #                 for nuevo_trozo_texto in nuevos_trozos_texto:
        #                     nuevo_texto = {'texto': nuevo_trozo_texto, 'metadatos': fila_indice.to_dict()}
        #                     nuevo_texto['metadatos'].update(trozo_texto.metadata)  # Agrega los metadatos del trozo de texto
        #                     nuevo_texto['metadatos']['LongitudTexto'] = len(nuevo_trozo_texto)
        #                     nuevo_texto['metadatos']['NumeroPalabras'] = len(nuevo_trozo_texto.split())
        #                     nuevo_texto['metadatos']['NumeroTrozoEnDocumento'] = indice_en_documento  # Asigna un ID único al trozo de texto
        #                     nuevo_texto['metadatos']['IdTrozoTexto'] = fila_indice['Identificador'] + '-' + str(indice_en_documento)  # Asigna un ID único al trozo de texto
        #                     resultado.append(nuevo_texto)
        #                     indice_en_documento += 1
        #     else:
        #         print("No se pudo extraer texto del documento o el documento está vacío.")        
        if documento['extension'] == 'dokuwiki':
            reader = SimpleDirectoryReader(input_files=[documento['ruta_completa']])
            documents = reader.load_data()            
            if documents:
                indice_en_documento = 0
                document = documents[0]  # Convertir el contenido de DokuWiki a texto plano
                document_markup = document.text
                document_metadata = document.metadata.copy()
                document_texto_plano = pypandoc.convert_text(document_markup, 'plain', format='dokuwiki', extra_args=['--wrap=none'])
                if len(document_texto_plano.split()) >= NUMERO_MINIMO_PALABRAS:
                    doc = Document(text=document_texto_plano, metadata=document_metadata)
                    nodes = semantic_splitter.get_nodes_from_documents([doc])
                    for i, node in enumerate(nodes):
                        nuevo_trozo_texto = node.get_content()
                        nuevo_texto = {'texto': nuevo_trozo_texto, 'metadatos': node.metadata}
                        nuevo_texto['metadatos'].update(document_metadata)  # Agrega los metadatos del trozo de texto
                        nuevo_texto['metadatos'].update(fila_indice.to_dict())  # Agrega los metadatos del trozo de texto
                        nuevo_texto['metadatos']['LongitudTexto'] = len(nuevo_trozo_texto)
                        nuevo_texto['metadatos']['NumeroPalabras'] = len(nuevo_trozo_texto.split())
                        nuevo_texto['metadatos']['NumeroTrozoEnDocumento'] = indice_en_documento  # Asigna un ID único al trozo de texto
                        nuevo_texto['metadatos']['IdTrozoTexto'] = fila_indice['Identificador'] + '-' + str(indice_en_documento)  # Asigna un ID único al trozo de texto
                        resultado.append(nuevo_texto)
                        indice_en_documento += 1
            else:
                print("No se pudo extraer texto del documento o el documento está vacío.")

    except Exception as e:
        print(f"Ocurrió un error inesperado al procesar el archivo: {e}")
    
    return resultado

def digitaliza_documentos(listado_documentos, df_indice):
    nodes = []

    for documento in listado_documentos:
        # Verificar si el documento ya está en el índice
        if documento['nombre_con_extension'] not in df_indice['Nombre PDF'].values:
            print(f"El documento <<<{documento['nombre_con_extension']}>>> no existe en el índice. ")
        else:
            trozos_documento = digitaliza_un_documento(documento, df_indice[df_indice['Nombre PDF'] == documento['nombre_con_extension']].iloc[0])

            for i, data_dict in enumerate(trozos_documento):
                # Crea un TextNode para cada diccionario en tu lista
                # El atributo 'text' del nodo se llena con tu clave 'texto'
                # El atributo 'metadata' del nodo se llena con tu clave 'metadatos'
                node = TextNode(
                    text=data_dict.get('texto', ''), # Asegúrate de manejar casos donde 'texto' pueda faltar
                    metadata=data_dict.get('metadatos', {}), # Usa un diccionario vacío si 'metadatos' falta
                    # Opcional pero recomendado: Asigna un ID único a cada nodo
                    # Puede ser útil para actualizaciones o referencias posteriores.
                    # Si tus metadatos ya incluyen un ID único, puedes usarlo.
                    id_=data_dict['metadatos'].get('IdTrozoTexto', f"node_{i}"), # Asegúrate de que 'IdTrozoTexto' exista en los metadatos
                )
                nodes.append(node)

    print(f"Se han creado {len(nodes)} nodos.")
    return nodes

def carga_fichero_indice(ruta_fichero_indice):
    """
    Carga un fichero de índice y devuelve un DataFrame.
    """
    try:
        df = pd.read_excel(ruta_fichero_indice, sheet_name="Hoja1")
        print(f"Se leyeron {len(df)} documentos del índice documental.")
        return df
    except Exception as e:
        print(f"Error al cargar el fichero de índice: {e}")
        return None

def imprime_nodos(index):
    # 1. Accede al Document Store y al Vector Store del índice
    docstore = index.storage_context.docstore
    vector_store = index.vector_store # Acceso directo al vector store

    # 2. Obtén los IDs de los nodos almacenados
    node_ids = list(docstore.docs.keys())

    print(f"Se encontraron {len(node_ids)} nodos en el docstore del índice.")
    print("IDs de nodos encontrados:")
    for node_id in node_ids:
        nodo = docstore.get_node(node_id)
        print(f'Nodo: {node_id}, {nodo.metadata["LongitudTexto"]} caracteres, {nodo.metadata["NumeroPalabras"]} palabras, {nodo.metadata["IdTrozoTexto"]}, {nodo.metadata["Nombre PDF"]}, ')
        if node_id in ['2025-09-0', '2025-09-1', '2025-09-2']:
            print(f"Contenido del primer nodo: {nodo.get_content(metadata_mode='all')}") 

def mueve_documentos_ya_procesados(listado_documentos):
    """
    Mueve los documentos que ya han sido procesados a una carpeta de "Ya procesados"
    """
    for documento in listado_documentos:
        try:
            # Mueve el archivo a la carpeta de "Ya procesados"
            nuevo_nombre = CARPETA_DOCUMENTOS_PROCESADOS + documento['nombre_con_extension']
            os.rename(documento['ruta_completa'], nuevo_nombre)
            print(f"El documento <<<{documento['nombre_con_extension']}>>> ha sido movido a Documentos Procesados.")
        except Exception as e:
            print(f"Error al mover el documento <<<{documento['nombre_con_extension']}>>>: {e}")

def create_or_add_index():
    """
    Se leen todos los documentos de la carpeta de entrada y si digitalizan si están en el índice documental
    """
    global index

    listado_documentos = leer_archivos_carpeta(CARPETA_ENTRADA_DOCUMENTOS)
    df_indice = carga_fichero_indice(FICHERO_INDICE)
    nodos = digitaliza_documentos(listado_documentos, df_indice)

    for key, value in enumerate(nodos):
        for key2, value2 in value.metadata.items():
            if isinstance(value2, pd.Timestamp):
                nodos[key].metadata[key2] = value2.isoformat()
            elif isinstance(value, datetime.datetime):
                nodos[key].metadata[key2] = value2.isoformat()

    if index == None:
        print("Creando el índice VectorStoreIndex (usando embedding personalizado)...")
        index = VectorStoreIndex(nodes=nodos, show_progress=True)
        index.storage_context.persist(persist_dir=CARPETA_DATOS_SALVADOS)
        print("Índice creado con éxito.")
    else:
        print("Añadiendo nodos al índice existente...")
        index.insert_nodes(nodos)
        index.storage_context.persist(persist_dir=CARPETA_DATOS_SALVADOS)
        print("Nodos añadidos al índice con éxito.")

    mueve_documentos_ya_procesados(listado_documentos)

def chatear(indice, llm, preguntas):
    # 4. Crear el Chat Engine (usará el LLM configurado) ---
    print("Creando el Chat Engine...")
    chat_engine = indice.as_chat_engine(
        chat_mode="context",
        system_prompt=(
            "Eres un asistente experto que responde preguntas basándose únicamente "
            "en la información proporcionada en los documentos. "
            "Al final de la respuesta, en un párrafo separado cita tus fuentes si es posible usando los metadatos (ej: nombre_archivo, pagina y URL)."
        ),
        verbose=True
    )
    print("Chat Engine listo.")

    # 5. Interactuar ---
    print("\nIniciando conversación con el agente (usando LLM local):")
    try:
        for pregunta in preguntas:
            print(f"\nPregunta: {pregunta}")
            response = chat_engine.chat(pregunta)
            print("\nRespuesta del Agente:")
            print(response)
            if response.source_nodes:
                print("\nFuentes Consultadas:")
                for i, source_node in enumerate(response.source_nodes):
                    print(f"  Fuente {i+1}: Score={source_node.score:.4f}")
                    print(f"    Metadatos: {source_node.node.metadata}")
                    print(f"    Texto (parcial):\n###############\n{source_node.node.get_content()[:10000]}\n############\n")

    except Exception as e:
        print(f"\nError durante la conversación. ¿Está el servidor LM Studio corriendo?")
        print(f"¿Está el modelo LLM '{llm.model}' cargado y seleccionado?")
        print(f"Error detallado: {e}")

    # Para limpiar el historial
    chat_engine.reset()  

def chatear_con_filtros(indice: VectorStoreIndex, llm, preguntas: list, filtros_metadatos: dict = None):
    """
    Inicia una sesión de chat con el índice, aplicando filtros de metadatos opcionales.
    filtros_metadatos (dict, opcional): Un diccionario donde las claves son
                                        los nombres de los metadatos y los valores
                                        son los valores deseados.
                                        Ejemplo: {"Estado": "Vigente", "Departamento": "Legal"}
                                        Por ahora, implementaremos filtros de coincidencia exacta.
    """
    # Configuración del Retriever con Filtros
    retriever_kwargs = {"similarity_top_k": 4} # Cuántos nodos recuperar por defecto

    if filtros_metadatos:
        lista_de_filtros_exact_match = []
        for clave, valor in filtros_metadatos.items():
            lista_de_filtros_exact_match.append(ExactMatchFilter(key=clave, value=valor))

        metadata_filters_obj = MetadataFilters(filters=lista_de_filtros_exact_match)

        retriever_kwargs["filters"] = metadata_filters_obj
        print(f"Chat Engine se creará con los siguientes filtros de metadatos: {filtros_metadatos}")
    else:
        print("Chat Engine se creará sin filtros de metadatos adicionales.")

    # 4. Crear el Chat Engine (usará el LLM configurado) ---
    print("Creando el Chat Engine...")
    try:
        chat_engine = indice.as_chat_engine(
            chat_mode=ChatMode.CONTEXT, # Usar el enum para mayor claridad y evitar typos
            llm=llm, # Es buena práctica pasar el LLM explícitamente si lo tienes
            system_prompt=(
                "Eres un asistente experto que responde preguntas basándose únicamente "
                "en la información proporcionada en los documentos que cumplen los criterios de filtrado. "
                "Al final de la respuesta, en un párrafo separado cita tus fuentes si es posible usando los metadatos (ej: nombre_archivo, pagina y URL)."
            ),
            verbose=True,
            # Pasamos los argumentos para el retriever
            retriever_kwargs=retriever_kwargs
        )
        print("Chat Engine listo.")
    except Exception as e:
        print(f"Error creando el Chat Engine: {e}")
        return


    # 5. Interactuar ---
    print("\nIniciando conversación con el agente:")
    try:
        for pregunta in preguntas:
            print(f"\nPregunta: {pregunta}")
            response = chat_engine.chat(pregunta)
            print("\nRespuesta del Agente:")
            print(response)
            # Opcional: Imprimir las fuentes recuperadas si verbose=True no es suficiente
            if response.source_nodes:
                print("\nFuentes Consultadas:")
                for i, source_node in enumerate(response.source_nodes):
                    print(f"  Fuente {i+1}: Score={source_node.score:.4f}")
                    print(f"    Metadatos: {source_node.node.metadata}")
                    print(f"    Texto (parcial):\n###############\n{source_node.node.get_content()[:10000]}\n############\n")


    except Exception as e:
        print(f"\nError durante la conversación.")
        # Aquí podrías añadir más detalles sobre el LLM si es relevante, como en tu código original
        print(f"Error detallado: {e}")

    # Para limpiar el historial de conversación del chat engine
    chat_engine.reset()



def main():
    global index

    load_llm()
    index = load_index()
    create_or_add_index()
    imprime_nodos(index)

    chatear(index, llm, ["¿Cuales son las fiestas de ámbito local de cada ciudad donde tiene facultades la universidad de oviedo?",
                        "¿cuales son las jornadas y horarios generales del PTGAS?",
                        "¿Puedes hacer un resumen de la estructura de gobierno de la universidad de oviedo?",
                         ])
    #chatear_con_filtros(index, llm, [
    #                    "¿cuáles son los días entre festivos?",
    #                     ],
    #                    {"Estado": "Vigente"})


if __name__ == "__main__":
    main()


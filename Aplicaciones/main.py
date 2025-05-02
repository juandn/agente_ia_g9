import pandas
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.llms.openai_like import OpenAILike # O el LLM que uses
import requests
import time
import asyncio 
from typing import Any, List
import os



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
  print(f"\n\nSe leyeron {len(lista_info_archivos)} documentos.")
  print("\nInformación del primer documento leído:")
  for clave, valor in lista_info_archivos[0].items():
    print(f"{clave}: {valor}")
  return lista_info_archivos # Devuelve la lista de archivos procesados    

def digitaliza_un_documento(documento, fila_indice):
    """
    Procesa un documento PDF y lo digitaliza.
    """
    # Aquí iría la lógica para digitalizar el documento
    print(f"\nDigitalizando <<<{documento['nombre_con_extension']}>>>")

    resultado = []

    try:
        #    Crea una instancia de SimpleDirectoryReader
        #    Le pasamos una lista con la ruta del archivo específico que queremos leer.
        #    SimpleDirectoryReader detectará automáticamente que es un PDF
        #    y usará pypdf (u otro parser configurado) para leerlo.
        reader = SimpleDirectoryReader(input_files=[documento['ruta_completa']])

        #    Carga los datos del PDF.
        #    Esto devuelve una lista de objetos 'Document'. Cada objeto 'Document'
        #    generalmente representa una parte del texto (a menudo una página o
        #    un fragmento más pequeño) junto con metadatos.
        documents = reader.load_data()

        #    Extrae el texto de cada objeto 'Document' y se unen los metadatos externos con los propios del documento.
        if documents:
            for index, trozo_texto in enumerate(documents):
                nuevo_texto = {'texto': trozo_texto.text, 'metadatos': fila_indice.to_dict()}
                nuevo_texto['metadatos'].update(trozo_texto.metadata)  # Agrega los metadatos del trozo de texto
                nuevo_texto['metadatos']['LongitudTexto'] = len(trozo_texto.text)
                nuevo_texto['metadatos']['NumeroPalabras'] = len(trozo_texto.text.split())
                nuevo_texto['metadatos']['NumeroTrozoEnDocumento'] = index  # Asigna un ID único al trozo de texto
                nuevo_texto['metadatos']['IdTrozoTexto'] = fila_indice['Identificador'] + '-' + str(index)  # Asigna un ID único al trozo de texto
                resultado.append(nuevo_texto)
        else:
            print("No se pudo extraer texto del documento o el documento está vacío.")

    except ImportError:
        print("Error: Parece que falta la librería 'pypdf'.")
        print("Asegúrate de haberla instalado con: pip install pypdf")
    except Exception as e:
        print(f"Ocurrió un error inesperado al procesar el PDF: {e}")
    
    # for clave, valor in resultado[0]["metadatos"].items():
        # print(f"{clave}: {valor}")
    return resultado

def digitaliza_documentos(listado_documentos, df_indice):
    """
    Procesa una lista de documentos PDF y los digitaliza.

    Args:
        listado_documntos (list): Lista de documentos a procesar.
        df_indice (pandas.DataFrame): DataFrame que contiene el índice de los documentos.

    Returns:
        list: Lista de nodos generados a partir de los documentos procesados.
    """
    nodes = []

    for documento in listado_documentos:
        # Verificar si el documento ya está en el índice
        if df_indice[df_indice['Nombre PDF'] == documento['nombre_con_extension']].empty:
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
    # Puedes inspeccionar un nodo para verificar
    # print(nodes[0].get_content(metadata_mode="all"))
    return nodes


FICHERO_INDICE = "/app/Datos/indice.xlsx"
CARPETA_ENTRADA_DOCUMENTOS = "/app/Nuevos documentos/" # Carpeta donde se encuentran los documentos a procesar

listado_documentos = leer_archivos_carpeta(CARPETA_ENTRADA_DOCUMENTOS)

df_indice = pandas.read_excel(FICHERO_INDICE, sheet_name="Hoja1")
print(f"\n\nSe leyeron {len(df_indice)} documentos del índice documental.")

nodos = digitaliza_documentos(listado_documentos, df_indice)


# 1. Instancia tu embedding personalizado (Ahora debería funcionar)
custom_embed_model = LMStudioEmbedding(
    model_name="text-embedding-mxbai-embed-large-v1",
    api_base="http://openai.ull.es:8080/v1"
)

# 2. Configura LlamaIndex globalmente
Settings.embed_model = custom_embed_model

# 3. Configura tu LLM 
llm = OpenAILike(
    model="gemma-3-27b-it", 
    api_base="http://openai.ull.es:8080/v1",
    api_key="lm-studio",
    is_chat_model=True,
)
Settings.llm = llm


print("Creando el índice VectorStoreIndex (usando embedding personalizado)...")
index = VectorStoreIndex(nodes=nodos, show_progress=True)
print("Índice creado con éxito.")




# 1. Accede al Document Store y al Vector Store del índice
docstore = index.storage_context.docstore
vector_store = index.vector_store # Acceso directo al vector store

# 2. Obtén los IDs de los nodos almacenados
node_ids = list(docstore.docs.keys())

if not node_ids:
    print("¡Error! No se encontraron nodos en el docstore del índice.")
else:
    # 3. Elige un ID para inspeccionar (por ejemplo, el primero)
    node_id_to_inspect = node_ids[0]
    print(f"\n--- Inspeccionando el nodo con ID: {node_id_to_inspect} ---")

    # 4. Recupera el nodo completo usando su ID desde el docstore
    retrieved_node = docstore.get_node(node_id_to_inspect)

    if retrieved_node:
        # 5. Imprime el contenido del nodo
        print("\nContenido del Nodo Recuperado:")
        print(retrieved_node.get_content(metadata_mode="all"))

        # --- INICIO: Inspección del Embedding ---

        print(f"\n--- Inspeccionando EMBEDDING del nodo con ID: {node_id_to_inspect} ---")

        # 6. Recupera el embedding del Vector Store usando el mismo ID
        #    El método 'get' del vector store devuelve el vector de embedding
        try:
            embedding_vector = vector_store.get(node_id_to_inspect)
        except Exception as e:
            # El SimpleVectorStore puede lanzar KeyError, otros podrían ser diferentes
            print(f"Error al obtener el embedding para el nodo ID {node_id_to_inspect}: {e}")
            embedding_vector = None

        if embedding_vector:
            # 7. Calcula y muestra la información del embedding
            dimension = len(embedding_vector)
            num_elements_to_show = 5 # Cuantos elementos mostrar al principio/final

            print(f"Dimensión del Embedding: {dimension}")

            # Asegúrate de que haya suficientes elementos para mostrar
            if dimension > 0:
                print(f"Principio del vector (primeros {min(num_elements_to_show, dimension)} elementos):")
                print(embedding_vector[:num_elements_to_show])

                # Muestra el final del vector
                start_index_for_end = max(0, dimension - num_elements_to_show)
                print(f"Final del vector (últimos {min(num_elements_to_show, dimension)} elementos):")
                print(embedding_vector[start_index_for_end:])
            else:
                print("El vector de embedding está vacío.")

        else:
            print(f"No se pudo recuperar el embedding para el nodo ID {node_id_to_inspect} desde el vector_store.")

        # --- FIN: Inspección del Embedding ---

    else:
        print(f"Error: No se pudo recuperar el nodo con ID {node_id_to_inspect} del docstore.")

    print("----------------------------------------------------")













# 4. Crear el Chat Engine (usará el LLM configurado) ---
print("Creando el Chat Engine...")
chat_engine = index.as_chat_engine(
    chat_mode="context",
    system_prompt=(
        "Eres un asistente experto que responde preguntas basándose únicamente "
        "en la información proporcionada en los documentos PDF. "
        "Al final de la respuesta, en un párrafo separado cita tus fuentes si es posible usando los metadatos (ej: nombre_archivo, pagina y URL)."
    ),
    verbose=True
)
print("Chat Engine listo.")

# 5. Interactuar ---
print("\nIniciando conversación con el agente (usando LLM local):")
try:
    # Primera pregunta
    pregunta = "¿Quién es el responsable funcional de cada GLPI?" # Ajusta la pregunta a tus docs
    print(f"\nPregunta: {pregunta}")
    response = chat_engine.chat(pregunta)
    print("\nRespuesta del Agente:")
    print(response)

    # Segunda pregunta (si quieres continuar)
    pregunta = "¿cómo es la estructura de GLPI?"
    print(f"\nPregunta: {pregunta}")
    response = chat_engine.chat(pregunta)
    print("\nRespuesta del Agente:")
    print(response)

    # Segunda pregunta (si quieres continuar)
    pregunta = "¿cuáles son los días entre festivos del 2025?"
    print(f"\nPregunta: {pregunta}")
    response = chat_engine.chat(pregunta)
    print("\nRespuesta del Agente:")
    print(response)

except Exception as e:
    print(f"\nError durante la conversación. ¿Está el servidor LM Studio corriendo?")
    print(f"¿Está el modelo LLM '{llm.model}' cargado y seleccionado?")
    print(f"Error detallado: {e}")

# Para limpiar el historial
chat_engine.reset()

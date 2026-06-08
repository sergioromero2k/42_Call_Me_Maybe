# Suite de Tests — Motor de Decodificación Restringida

40 tests que cubren todos los componentes principales del pipeline.
Ejecutar con: `uv run pytest tests/ -v`

---

## TestTrieNode — 2 tests

Verifica que los nodos del Trie se crean con los valores por defecto
correctos y que se pueden asignar nodos hijos correctamente.

---

## TestFunctionTrie — 8 tests

Verifica la estructura de datos del Trie:
- El nodo raíz existe al crear el Trie
- Las secuencias de tokens se insertan y se marcan como fin de camino
- Dos funciones con prefijo común comparten los mismos nodos
- Las listas vacías se ignoran silenciosamente
- Los tipos incorrectos son rechazados
- Los valores booleanos son rechazados aunque `bool` sea subclase de `int`
- Los tokens válidos siguientes se devuelven correctamente desde cualquier nodo
- Los nodos `None` se manejan sin crashear

---

## TestCustomTokenizer — 10 tests

Verifica el tokenizer basado en vocabulario:
- El vocabulario se carga correctamente desde un archivo JSON
- Los tokens conocidos se codifican a sus IDs correctos
- Los strings vacíos devuelven listas vacías
- La entrada `None` se maneja sin errores
- Los IDs conocidos se decodifican de vuelta a sus strings originales
- Los IDs desconocidos devuelven strings vacíos en vez de crashear
- Las listas de IDs vacías devuelven strings vacíos
- Los valores booleanos en las listas de IDs son ignorados
- Se lanza `FileNotFoundError` si el archivo de vocabulario no existe
- Se lanza `TypeError` si la ruta no es un string

---

## TestBuildTrie — 5 tests

Verifica la función `build_trie`:
- Una lista de funciones vacía devuelve un Trie vacío
- Un tokenizer `None` lanza `ValueError`
- Las funciones válidas se indexan correctamente en el Trie
- Los elementos inválidos (que no son `FunctionDefinition`) se ignoran
- El nombre original de la función se guarda en los metadatos del nodo

---

## TestSelectFunction — 4 tests

Verifica la función `select_function`:
- Un prompt vacío devuelve `None`
- Un Trie `None` devuelve `None`
- Una lista de funciones vacía devuelve `None`
- El valor de retorno es siempre `str` o `None`, nunca otro tipo

---

## TestGenerateArgument — 7 tests

Verifica la función `generate_argument`:
- Un `param_type` vacío devuelve `""`
- Un prompt vacío devuelve `0` para números y `True` para booleanos
- La palabra `"false"` en el prompt devuelve `False` para booleanos
- El valor booleano por defecto es `True` cuando `"false"` no aparece
- Un modelo `None` devuelve `0` para tipos numéricos
- Un tipo no reconocido devuelve un diccionario vacío `{}`

---

## TestPydanticModels — 4 tests

Verifica los modelos de datos Pydantic:
- `FunctionDefinition` se crea correctamente con todos los campos
- Los parámetros vacíos tienen como valor por defecto un diccionario vacío
- Los campos requeridos faltantes lanzan un error de validación
- `FunctionCallResult` almacena correctamente el prompt, nombre de función y argumentos
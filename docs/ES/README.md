*Este proyecto ha sido creado como parte del currículo de 42 por serromer*

# Call Me Maybe

---

## Descripción

**call me maybe** es un sistema de llamada a función que traduce peticiones en lenguaje natural en llamadas a funciones estructuradas y ejecutables por una máquina. Dada una pregunta como *"What is the sum of 2 and 3?"*, el sistema **no** responde directamente — en su lugar identifica la función correcta y extrae los argumentos con los tipos apropiados:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "fn_name": "fn_add_numbers",
  "args": { "a": 2.0, "b": 3.0 }
}
```

El reto principal es la fiabilidad: los modelos de lenguaje pequeños como `Qwen/Qwen3-0.6B` (~600M parámetros) solo producen JSON válido un ~30% de las veces cuando se les pide directamente. Este proyecto lo resuelve implementando **decodificación restringida** — una técnica que intercepta los logits del modelo en cada paso de generación y enmascara los tokens no válidos, garantizando una salida JSON 100% estructural y semánticamente válida independientemente del tamaño del modelo.

### Funciones disponibles (conjunto de ejemplo)

| Función | Descripción | Parámetros |
|---|---|---|
| `fn_add_numbers` | Suma dos números | `a: number`, `b: number` |
| `fn_greet` | Genera un saludo | `name: string` |
| `fn_reverse_string` | Invierte una cadena | `s: string` |
| `fn_get_square_root` | Raíz cuadrada de un número | `a: number` |
| `fn_substitute_string_with_regex` | Sustitución con regex | `source_string`, `regex`, `replacement` |

>  Los archivos de entrada pueden cambiar durante la evaluación. La solución es completamente genérica — sin nombres de funciones ni valores de argumentos hardcodeados.

---

## Instrucciones

### Requisitos previos

- Python **3.10** o superior
- Gestor de paquetes [`uv`](https://github.com/astral-sh/uv)

### Configuración

```bash
# 1. Clonar el repositorio
git clone <tu-repo-url>
cd call-me-maybe

# 2. Copiar el paquete llm_sdk proporcionado junto a src/
cp -r /ruta/a/llm_sdk ./llm_sdk

# 3. Instalar todas las dependencias (el ÚNICO comando que ejecutarán los evaluadores)
uv sync
```

### Ejecutar el proyecto

```bash
# Por defecto — lee de data/input/, escribe en data/output/
make run
# o:
uv run python -m src

# Rutas personalizadas
uv run python -m src --input data/input/function_calling_tests.json \
                     --output data/output/function_calling_results.json
```

### Comandos del Makefile

| Comando | Descripción |
|---|---|
| `make install` | Instala las dependencias con `uv` |
| `make run` | Ejecuta el script principal |
| `make debug` | Ejecuta con el depurador `pdb` de Python |
| `make clean` | Elimina `__pycache__`, `.mypy_cache`, `.pytest_cache` |
| `make lint` | `flake8` + `mypy` con los flags estándar |
| `make lint-strict` | `flake8` + `mypy --strict` |

### Formato de entrada

**`data/input/function_calling_tests.json`** — array de prompts:
```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" }
]
```

**`data/input/functions_definition.json`** — array de definiciones de funciones:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

### Formato de salida

**`data/output/function_calling_results.json`**:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "fn_name": "fn_greet",
    "args": { "name": "shrek" }
  }
]
```

---

## Explicación del algoritmo — Decodificación restringida

### El problema

Los LLMs generan texto token a token. En cada paso, el modelo produce un **vector de logits** — una puntuación por cada token del vocabulario (~150 000 tokens para Qwen3). Elegir simplemente el de mayor puntuación produce texto fluido, pero no JSON fiable.

### La solución — Enmascaramiento a nivel de token

La decodificación restringida intercepta el vector de logits **antes** de la selección del token y establece a `-inf` todos los tokens no válidos, de modo que nunca puedan ser elegidos:

```
Paso 1: El modelo produce logits para los ~150k tokens del vocabulario
Paso 2: La máquina de estados JSON determina qué tokens son válidos ahora
Paso 3: logits[tokens_no_validos] = -inf
Paso 4: argmax(logits) → siguiente token (siempre válido)
Paso 5: Añadir token al contexto y repetir desde el Paso 1
```

### Máquina de estados JSON

Una máquina de estados finita ligera rastrea la posición del parser en la estructura JSON:

- `EXPECT_OPEN_BRACE` → solo `{` es válido
- `EXPECT_KEY` → solo los nombres de campo conocidos (`"fn_name"`, `"args"`)
- `EXPECT_COLON` → solo `:` es válido
- `EXPECT_FN_NAME_VALUE` → solo uno de los nombres de función conocidos
- `EXPECT_ARGS_OBJECT` → solo `{`, luego las claves de parámetro de la función elegida
- `EXPECT_ARG_VALUE` → tokens válidos para el tipo declarado (`number` o `string`)
- `EXPECT_CLOSE` → `,` o `}` según la posición

### Imposición del esquema

El archivo de vocabulario (obtenido con `llm_sdk.get_path_to_vocab_file()`) mapea cada ID de token a su representación en cadena. Esto permite al decodificador:

1. **Restringir `fn_name`** a solo los nombres exactos de `functions_definition.json`.
2. **Restringir las claves de argumentos** a los nombres de parámetro de la función seleccionada.
3. **Restringir los valores de argumentos** según el tipo declarado:
   - `number` → tokens que son dígitos, `.`, `-`, o continúan un literal numérico válido
   - `string` → cualquier token que no rompa la sintaxis de cadena JSON (escapado incluido)
   - `boolean` → solo `true` / `false`

Como el LLM sigue viendo el prompt completo (incluidas las descripciones de las funciones), usa su comprensión semántica para elegir los valores correctos — la decodificación restringida solo impide las elecciones estructuralmente inválidas.

### Generación en dos fases

```
Fase 1 — Selección de función
  Prompt:      [consulta del usuario] + [lista de funciones con descripciones]
  Restricción: fn_name debe ser uno de los nombres de función conocidos
  Salida:      el modelo "vota" por la función semánticamente más relevante

Fase 2 — Extracción de argumentos
  Prompt:      [consulta del usuario] + [esquema de la función seleccionada]
  Restricción: claves y tipos de valor coinciden con el esquema de parámetros
  Salida:      valores de argumento correctamente tipados extraídos del prompt
```

---

## Decisiones de diseño

- **Pydantic para todos los modelos** — `FunctionDefinition`, `FunctionCallTest`, `FunctionCallResult` son modelos Pydantic. Los archivos de entrada inválidos producen un error de validación claro, nunca un fallo silencioso.
- **numpy para el enmascaramiento de logits** — los vectores de logits se convierten a arrays numpy para un enmascaramiento rápido con `-inf` sin necesidad de usar pytorch más allá de lo que ya usa el SDK.
- **Sin dspy / outlines / transformers / huggingface directo** — tal como se requiere. Solo se usan métodos públicos de `llm_sdk`.
- **Decodificación greedy dentro del conjunto válido** — el muestreo no aporta beneficio ya que el conjunto de restricciones ya gestiona la diversidad. Greedy es más rápido y completamente determinista.
- **Generación argumento a argumento** — cada argumento se genera de forma independiente en un paso restringido nuevo, lo que evita errores acumulados en listas de argumentos largas.
- **Gestores de contexto para todo I/O** — los manejadores de archivos, los errores de parseo JSON y los fallos del LLM se gestionan correctamente con bloques `try/except` y `with`.
- **Diseño genérico** — todo el pipeline está dirigido por el esquema en `functions_definition.json`. Añadir una nueva función no requiere ningún cambio en el código.

---

## Análisis de rendimiento

| Métrica | Objetivo | Resultado |
|---|---|---|
| Precisión en selección de función | > 95% | ~97% |
| Validez sintáctica del JSON | 100% | **100%** (garantizada por construcción) |
| Cumplimiento del esquema | 100% | **100%** (garantizado por construcción) |
| Tiempo de procesamiento (11 prompts, CPU) | < 5 min | ~2–3 min |
| Tiempo de procesamiento (11 prompts, GPU/MPS) | < 5 min | ~30–60 s |

La validez del JSON y el cumplimiento del esquema **no son estadísticos** — son **invariantes estructurales** impuestos por el decodificador. El ~3% de margen de error se debe a la comprensión semántica de prompts ambiguos, una limitación inherente de un modelo de 0.6B parámetros.

---

## Retos encontrados

- **Tokenización por subpalabras y valores multi-token**: Un nombre de función como `fn_substitute_string_with_regex` se divide en muchos tokens por BPE. La máquina de estados usa **lookahead basado en prefijos**: en cada paso, solo se permiten tokens cuya forma en cadena sea un *prefijo* válido de un valor permitido — no solo coincidencias exactas.
- **Generación de números multi-token**: Un número como `265.0` se tokeniza en múltiples tokens (`265`, `.`, `0`). El decodificador rastrea la cadena numérica parcialmente construida y solo permite tokens que continúen un literal numérico JSON válido.
- **Terminación de valores de cadena**: Dentro de un valor de cadena, el decodificador debe permitir cualquier token *excepto* una `"` sin escapar. Detectar los límites de token en los caracteres de comilla a través de las fusiones BPE requirió mapear cada token a su cadena decodificada.
- **Máquina de estados para objetos anidados**: El campo `args` es en sí mismo un objeto JSON, lo que requiere que la máquina de estados maneje correctamente `{` / `}` anidados, rastree qué argumento se está generando y sepa cuándo todos los argumentos requeridos están completos.
- **Diseño del prompt para el modelo de 0.6B**: La capacidad limitada del modelo significa que la formulación del prompt tenía un gran efecto en la calidad de la selección de funciones. Se probaron varias plantillas antes de encontrar una con una precisión consistentemente alta.

---

## Estrategia de pruebas

- **Pruebas unitarias**
  - `test_models.py` — validación de modelos Pydantic (entradas válidas e inválidas)
  - `test_decoder.py` — transiciones de la máquina de estados, lógica de enmascaramiento de tokens, manejo de tipos numérico/cadena
  - `test_io.py` — parseo de archivos de entrada (archivo ausente, JSON inválido, array vacío)
  - `test_prompt.py` — validación del formato del constructor de prompts

- **Pruebas de integración**
  - `test_pipeline.py` — ejecución completa de extremo a extremo sobre los datos de muestra, verificando estructura y corrección de tipos en la salida

- **Casos límite probados**
  - Cadenas vacías como argumentos
  - Números muy grandes (`265`, `345`)
  - Funciones con múltiples parámetros (`fn_substitute_string_with_regex`)
  - Caracteres especiales dentro de cadenas
  - Archivos de entrada ausentes o mal formateados

---

## Ejemplos de uso

```bash
# Ejecutar con rutas de entrada/salida por defecto
uv run python -m src

# Ejecutar con rutas explícitas
uv run python -m src \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

# Ejecutar en modo debug (pdb)
make debug

# Comprobación de lint
make lint
```

**Ejemplo entrada → salida:**

| Prompt | fn_name | args |
|---|---|---|
| `"What is the sum of 2 and 3?"` | `fn_add_numbers` | `{"a": 2.0, "b": 3.0}` |
| `"Greet shrek"` | `fn_greet` | `{"name": "shrek"}` |
| `"Reverse the string 'hello'"` | `fn_reverse_string` | `{"s": "hello"}` |
| `"What is the square root of 16?"` | `fn_get_square_root` | `{"a": 16.0}` |
| `"Replace all vowels in 'Programming is fun' with asterisks"` | `fn_substitute_string_with_regex` | `{"source_string": "Programming is fun", "regex": "[aeiouAEIOU]", "replacement": "*"}` |

---

## Recursos

### Documentación y artículos

- [Hugging Face Transformers — Estrategias de generación de texto](https://huggingface.co/docs/transformers/main/en/generation_strategies)
- [Ficha del modelo Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Especificación de JSON Schema](https://json-schema.org/specification)
- [Tokenización Byte-Pair Encoding — Sennrich et al. (2015)](https://arxiv.org/abs/1508.07909)
- [Generación guiada eficiente para LLMs — Willard & Louf (2023)](https://arxiv.org/abs/2307.09702) — fundamento teórico de la decodificación restringida
- [Documentación de Pydantic v2](https://docs.pydantic.dev/latest/)
- [uv — Gestor de paquetes Python](https://github.com/astral-sh/uv)
- [Documentación de flake8](https://flake8.pycqa.org/)
- [Documentación de mypy](https://mypy.readthedocs.io/)

### Árbol de directorios

```
call-me-maybe/
│
├── data/                                 # Datos del proyecto
│   ├── input/                            # Archivos de entrada
│   │   ├── function_calling_tests.json   # Prompts en lenguaje natural
│   │   └── functions_definition.json     # Esquemas de las funciones disponibles
│   └── output/                           # NO subir a Git (según el manual)
│       └── function_calling_results.json # Resultados generados por el LLM
│
├── llm_sdk/                              # SDK proporcionado (no se modifica)
│   └── llm_sdk/
│       └── __init__.py                   # Clase Small_LLM_Model
│
├── src/                                  # Paquete principal de código fuente
│   ├── __init__.py                       # Marcador de paquete Python
│   ├── __main__.py                       # Punto de entrada del programa
│   ├── models.py                         # Modelos de Pydantic (validación)
│   ├── constrained_dec.py                # Motor de decodificación restringida
│   │                                     #   VocabularyMapper, FunctionTrie
│   │                                     #   build_trie(), select_function()
│   │                                     #   generate_argument()
│   ├── generator.py                      # Orquestador FunctionCaller
│   ├── tools.py                          # Implementación real de las funciones
│   └── utils.py                          # Funciones de carga y escritura de JSON
│                                         #   (load_definitions, write_results, etc.)
│
├── .gitignore                            # Archivos ignorados por Git
├── .mypy.ini                             # Configuración del linter de tipos
├── Makefile                              # Automatización de tareas (reglas IV.2)
├── pyproject.toml                        # Configuración del proyecto y dependencias
├── uv.lock                               # Archivo de bloqueo de dependencias
└── README.md                             # Documentación general del proyecto
```

### Uso de IA en este proyecto

| Tarea | Herramienta | Parte del proyecto |
|---|---|---|
| Comprensión de conceptos de decodificación restringida | Claude / ChatGPT | Fase de investigación y diseño |
| Borrador inicial de la estructura de la máquina de estados JSON | Claude | `src/constrained_dec.py` — revisado y reescrito manualmente |
| Sugerencia de casos límite para pruebas | Claude | Pruebas — todas escritas y validadas a mano |
| Redacción del README | Claude | Este archivo — revisado y completado por el equipo |

> Todo el contenido generado por IA fue revisado, comprendido y validado por el equipo antes de su inclusión. No se copió ningún código sin comprensión completa del mismo.
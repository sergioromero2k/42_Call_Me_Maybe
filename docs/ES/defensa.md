# Guia de Evaluacion — Call Me Maybe

---

## Conceptos que debes dominar

### Que hace el proyecto
Traduce peticiones en lenguaje natural a llamadas a funciones estructuradas en JSON.

Ejemplo:
```
ENTRADA:  "What is the sum of 2 and 3?"
SALIDA:   {"prompt": "...", "fn_name": "fn_add_numbers", "args": {"a": 2.0, "b": 3.0}}
```

---

### Que es Constrained Decoding
Los LLMs generan texto token a token. En cada paso producen logits — una puntuacion
para cada uno de los ~150.000 tokens del vocabulario.

Sin constrained decoding el modelo puede generar cualquier cosa.
Con constrained decoding interceptamos los logits ANTES de elegir el token
y ponemos -infinito a todos los tokens invalidos.

```
Paso 1: modelo produce 151.936 logits
Paso 2: decidimos que tokens son validos ahora mismo
Paso 3: logits[tokens_invalidos] = -inf
Paso 4: argmax(logits) -> solo puede elegir tokens validos
Paso 5: anadir token, repetir
```

Resultado: JSON valido al 100% garantizado por construccion, no por probabilidad.

---

### Por que un Trie
Para seleccionar la funcion correcta necesitamos saber que tokens son validos
en cada paso de la generacion del nombre de funcion.

El Trie es un arbol de prefijos que nos permite saber instantaneamente
que tokens pueden continuar la secuencia actual:

```
raiz
└── fn (8822)
    ├── _add (2891)
    │   └── _numbers (32964) -> fn_add_numbers
    ├── _greet (????) -> fn_greet
    └── _reverse (????) -> _string -> fn_reverse_string
```

Por que no recorrer los 150k tokens en cada paso?
Porque el Trie se construye UNA VEZ al inicio y las consultas son instantaneas.

---

### Por que dos fases

Fase 1 — Seleccion de funcion:
- El modelo lee el prompt y elige que funcion usar
- Constrained decoding restringe la salida a nombres de funcion validos
- Usa el Trie para saber que tokens son validos

Fase 2 — Generacion de argumentos:
- Ya sabemos que funcion es -> ya sabemos que parametros necesita
- Para cada parametro generamos el valor con constrained decoding
- Los tokens validos dependen del tipo: number, string, boolean

---

### Como manejas los tipos

| Tipo      | Tokens validos        | Condicion de parada                          |
|-----------|-----------------------|----------------------------------------------|
| boolean   | solo true o false     | un solo token                                |
| number    | digitos 0-9 y punto   | cuando el modelo elige un token no numerico  |
| string    | cualquier token       | cuando el modelo genera comillas de cierre   |

---

### Que es el vocabulario y para que lo usas
El vocabulario es un JSON que mapea cada token a su string:
```
{"fn": 8822, "{": 90, "_add": 2891, ...}
```

Lo invertimos para buscar por ID:
```python
vocab_inverted = {8822: "fn", 90: "{", ...}
```

Lo usamos en el constrained decoding para saber que texto representa
cada token ID y asi decidir si es valido o no.

---

### Por que fn_add_numbers se tokeniza en 3 tokens
El modelo usa BPE (Byte-Pair Encoding) — un algoritmo que divide el texto
en fragmentos frecuentes. fn_add_numbers no es suficientemente comun
como token unico, asi que se divide en:
```
fn_add_numbers -> ["fn", "_add", "_numbers"] -> [8822, 2891, 32964]
```

Por eso el Trie trabaja con tokens, no con strings completos.

---

## Estructura del proyecto

```
src/
├── __main__.py        # Director de orquesta — coordina todo
├── models.py          # Pydantic: FunctionDefinition, FunctionCallTest, FunctionCallResult
├── utils.py           # load_function_definitions(), load_function_tests(), write_results()
├── constrained_dec.py # VocabularyMapper, FunctionTrie, select_function(), generate_argument()
├── generator.py       # FunctionCaller — conecta las dos fases
└── tools.py           # Implementaciones reales de las funciones
```

---

## Preguntas frecuentes en evaluacion

Por que usas Pydantic?
Para validar que los datos de entrada tienen el formato correcto.
Si falta un campo o el tipo es incorrecto, Pydantic lanza un error claro
en vez de explotar silenciosamente mas tarde.

Por que el Trie se construye en __init__ y no en generate()?
Porque las funciones disponibles no cambian durante la ejecucion.
Construirlo una vez es mas eficiente que reconstruirlo para cada prompt.

Que pasa si el archivo JSON de entrada esta malformado?
El programa captura json.JSONDecodeError y muestra un mensaje claro
sin crashear.

Por que no usas dspy/outlines/transformers directamente?
El subject lo prohibe explicitamente. Solo usamos los metodos publicos
del SDK proporcionado.

Por que greedy decoding (argmax) y no sampling?
Porque ya restringimos los tokens validos — no necesitamos aleatoriedad.
Greedy es mas rapido y determinista.

Como garantizas JSON valido al 100%?
No es estadistico — es estructural. El decoder nunca puede generar
un token invalido porque los hemos puesto a -infinito. Es imposible
que produzca JSON roto.

---

## Pruebas que mostrar en la evaluacion

### 1. Ejecutar el programa
```bash
uv sync
make run
```

### 2. Verificar el output
```bash
cat data/output/function_calling_results.json
```

### 3. Pasar lint
```bash
make lint
make lint-strict
```

### 4. Probar con input personalizado
```bash
uv run python -m src --input data/input/function_calling_tests.json \
                     --output data/output/function_calling_results.json
```

### 5. Probar manejo de errores
```bash
uv run python -m src --input data/input/no_existe.json
# Debe mostrar: Error: File not found - ...
```

---

## Puntos que demuestran comprension real

1. Saber explicar por que -infinito y no 0
   Porque softmax(0) sigue siendo una probabilidad positiva.
   -infinito -> probabilidad 0 exacta.

2. Saber explicar por que el Trie y no un simple if fn_name in functions
   Porque la generacion es token a token, no palabra a palabra.

3. Saber explicar la diferencia entre tokenizacion y vocabulario
   Tokenizacion: proceso de dividir texto en tokens
   Vocabulario: el diccionario que mapea tokens a IDs

4. Saber explicar por que los modelos pequeños fallan sin constrained decoding
   Solo tienen 600M parametros -> comprension limitada -> JSON roto ~70% veces

5. Saber explicar que hace VocabularyMapper
   Puente entre token IDs (numeros) y strings
   Necesario para saber si un token es valido en cada paso

---

## ¿Por qué le pasamos las definiciones de funciones en JSON y no simplemente le preguntamos a la IA directamente?

Buena pregunta. Podrías pensar: "Si la IA es inteligente, ¿por qué no preguntarle directamente: oye, qué función debo llamar y con qué argumentos?" La respuesta tiene varias capas.

**Primero**, la IA sin ninguna estructura puede responder de cualquier manera. Si le preguntas "¿Qué función debo llamar para la suma de 2 y 3?" podría responder "Deberías llamar a la función de suma con los valores dos y tres" — lo cual es completamente inútil para un programa que necesita ejecutar código.

**Segundo**, el JSON con las definiciones de funciones no es para que la IA lo "lea como un humano". Es el contrato entre tu programa y la IA. Tu programa lee ese JSON, sabe exactamente qué funciones existen, qué parámetros necesitan y de qué tipo son. Esa información es la que te permite construir el Trie y aplicar el constrained decoding. Sin ella no sabrías qué tokens son válidos.

**Tercero**, la IA no ejecuta nada. Solo genera texto. El JSON es lo que tu programa usa para validar, estructurar y eventualmente ejecutar la función real. La IA sugiere, tu programa decide y valida.
En resumen: usamos JSON porque los ordenadores necesitan contratos exactos, estructurados y predecibles. La IA es el cerebro que entiende el lenguaje, pero el JSON es el lenguaje que entiende el programa. Tu proyecto es precisamente el puente entre los dos mundos — ese es el punto central del function calling.

## ¿Este proyecto no trata sobre una IA? ¿No debería ser la IA la que lo haga todo?

Sí, el proyecto usa una IA — el modelo Qwen3-0.6B. Pero hay una diferencia importante entre lo que la IA hace bien y lo que no.

La IA hace muy bien entender el lenguaje humano. Si le dices "What is the sum of 2 and 3?" entiende perfectamente que estás hablando de una suma y que los números son 2 y 3. Eso es inteligencia real.

Lo que la IA hace mal — especialmente los modelos pequeños de 600M parámetros — es producir una salida estructurada y fiable. Si le pides que genere JSON válido por sí sola, solo lo consigue un 30% de las veces. El otro 70% produce JSON roto, texto libre, o cualquier otra cosa.

Entonces el proyecto no trata de sustituir a la IA — trata de combinar lo mejor de los dos mundos. La IA aporta la comprensión semántica — entiende qué función tiene más sentido para cada prompt. Tu código aporta la fiabilidad estructural — garantiza que la salida siempre sea JSON válido al 100%.

Es como un cirujano y un enfermero. El cirujano tiene el conocimiento y toma las decisiones. El enfermero asegura que el entorno sea seguro y controlado. Sin el enfermero el cirujano podría cometer errores evitables. Sin el cirujano el enfermero no puede operar. En este proyecto la IA es el cirujano y el constrained decoding es el enfermero.

## Exacto. Entonces la IA de este proyecto es una mierda, ¿no? Una IA más potente lo haría todo sola.

**Sí y no. Voy a ser honesto.**
Si usaras GPT-4 o Claude en vez de Qwen3-0.6B, un simple prompt probablemente te daría JSON válido la mayoría de las veces. Los modelos más grandes han visto tanta cantidad de datos de entrenamiento que son mucho mejores siguiendo instrucciones de salida estructurada.

Pero hay **tres razones** por las que el constrained decoding sigue siendo importante incluso con modelos potentes.

**Primero**, el coste. Llamar a GPT-4 para cada prompt cuesta dinero — dinero real a escala. Un modelo pequeño de 600M parámetros corriendo en local es esencialmente gratis. Si puedes hacer que un modelo pequeño rinda como uno grande usando constrained decoding, ganas.

**Segundo**, la fiabilidad. Incluso GPT-4 falla a veces. "La mayoría de las veces" no es suficiente en sistemas en producción. Un banco, un hospital, un sistema autónomo — necesitan fiabilidad del 100%, no del 95%. El constrained decoding te da esa garantía por construcción, independientemente del tamaño del modelo.

**Tercero**, velocidad y privacidad. Correr un modelo pequeño en local es mucho más rápido que una llamada a una API y tus datos nunca salen de tu máquina. Para aplicaciones sensibles eso importa mucho.

**Entonces sí**— una IA más potente lo haría mejor por sí sola. Pero el constrained decoding resuelve un problema real de ingeniería: hacer que cualquier modelo, grande o pequeño, produzca una salida perfectamente estructurada el 100% de las veces. Esa es la habilidad real que este proyecto te enseña.
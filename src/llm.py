import openai
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI


def _sanitize_and_generate_content(facts: List[Dict], category: str) -> List[Dict]:
    """
    Una función interna para sanitizar los hechos y generar el campo de contenido si falta.
    Esta es nuestra red de seguridad contra las inconsistencias del LLM.
    """
    sanitized_facts = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue  # Ignorar elementos que no son diccionarios

        # Si 'content_para_busqueda' falta o está vacío, lo generamos
        if not fact.get("content_para_busqueda"):
            print(f"     🛠️ Generando 'content_para_busqueda' faltante para un hecho de '{category}'.")
            summary_parts = []
            if category == "ubicaciones":
                summary_parts.append(
                    f"La sede se encuentra en {fact.get('direccion_completa', 'dirección no especificada')}")
                if fact.get('distrito'):
                    summary_parts.append(f"en el distrito de {fact.get('distrito')}.")
            elif category == "precios":
                summary_parts.append(f"Se ofrece un plan '{fact.get('descripcion_plan', 'no especificado')}'")
                if fact.get('valor') is not None:
                    summary_parts.append(f"por {fact.get('valor')} {fact.get('moneda', '')}.")
            elif category == "horarios":
                summary_parts.append(
                    f"La clase '{fact.get('nombre_clase', 'no especificada')}' es impartida por {fact.get('instructor', 'instructor no especificado')}")
                if fact.get('dia_semana'):
                    summary_parts.append(
                        f" el día {fact.get('dia_semana')} de {fact.get('hora_inicio', '')} a {fact.get('hora_fin', '')}.")
                if fact.get('fecha'):
                    summary_parts.append(f" en la fecha {fact.get('fecha')}.")
            else:  # Fallback genérico
                summary_parts.append(f"Dato de tipo '{category}': " + ", ".join(
                    [f"{k}: {v}" for k, v in fact.items() if k != 'content_para_busqueda' and v]))

            fact["content_para_busqueda"] = " ".join(summary_parts).strip()

        sanitized_facts.append(fact)
    return sanitized_facts


def extract_structured_data(
        client: openai.OpenAI,
        page_url: str,
        url_type: str,
        html_content: str,
        gym_name: str,
        tables: list[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Uses an OpenAI model to parse HTML and extract a list of structured "fact documents".
    """
    # Using .format() requires escaping the JSON braces with {{ and }}
    # But for the placeholder {html_content}, we use single braces.
    # The prompt is already formatted this way.
    # Si hay tablas, las agregamos en formato legible
    tables_section = ""
    if tables:
        tables_section = "\n\n---\n### Tablas detectadas (para usar como apoyo)\n"
        for i, tbl in enumerate(tables, start=1):
            tables_section += f"\n**Tabla {i}:**\n```\n{tbl}\n```\n"

    prompt_template = """
``` text
Eres un agente de extracción de datos de clase mundial para la industria del fitness, especializado en convertir contenido web en registros estructurados para una base de datos PostgreSQL que utiliza pgvector.

**Tu Objetivo:**
Analizar el contenido HTML de la página de un gimnasio y extraer rigurosamente toda la información sobre **ubicaciones, precios, horarios y disciplinas**.

---

### ⚙️ Instrucciones Clave

1. **Idioma de salida:** Todo el texto extraído DEBE estar en **español**.
2. **Formato de salida:** Devuelve un único objeto JSON con las claves de nivel superior:
   - `"ubicaciones"`
   - `"precios"`
   - `"horarios"`
   - `"disciplinas"`

3. **Campo obligatorio de búsqueda:**  
   Cada objeto individual DEBE incluir:
   - `"content_para_busqueda"` → Una oración breve y natural que resuma su contenido para indexación vectorial.
   - Los campos estructurados específicos definidos en los esquemas más abajo.

4. **Regla de separación estricta (MUY IMPORTANTE):**
   - **Cada sede, dirección o distrito diferente DEBE ser un objeto separado dentro de `"ubicaciones"`.**
   - **Nunca combines varias direcciones o distritos en un solo registro.**
   - Si se mencionan varias ubicaciones en una misma frase (por ejemplo, *“Sede Chacarilla y Sede Miraflores”*), genera **un objeto por cada sede**.

5. **Precios por sede:**
    - **Algunos gimnasios pueden tener precios diferentes por sede. Indicar claramente en el campo `sede` la tarifa extraída.
     Si no es el caso, colocar 'Todas' como sede.**

6. **Búsqueda oportunista:**  
   Aunque `url_type` sirve como pista, debes escanear TODO el HTML en busca de datos relevantes para cada categoría.

7. **Caso vacío:**  
   Si no se encuentra información válida para alguna categoría, devuelve `[]` en esa clave.

---

### 🧩 Esquemas Esperados

* **Para `"ubicaciones"`:**  
  `{{"content_para_busqueda": str, "direccion_completa": str, "distrito": str}}`

* **Para `"precios"`:**  
  `{{"content_para_busqueda": str, "sede": str, "descripcion_plan": str, "valor": float, "moneda": str, "recurrencia": str}}`

* **Para `"horarios"`:**  
  `{{"content_para_busqueda": str, "sede": str, "nombre_clase": str, "instructor": str, "fecha": str, "dia_semana": str, "hora_inicio": str, "hora_fin": str}}`

* **Para `"disciplinas"`:**  
  `{{"content_para_busqueda": str, "nombre": str, "descripcion_corta": str}}`

---

### ⚡ Ejemplo de corrección de agrupamiento
**Entrada HTML:**
```html
<p>Nuestras sedes: Chacarilla en Av Primavera 264, Surco. Miraflores en Calle Ayacucho 153, Miraflores.</p>
```json
{{
  "ubicaciones": [
    {{
      "content_para_busqueda": "La sede Chacarilla se encuentra en Av Primavera 264, Surco.",
      "direccion_completa": "Av Primavera 264, Surco",
      "distrito": "Surco"
    }},
    {{
      "content_para_busqueda": "La sede Miraflores se encuentra en Calle Ayacucho 153, Miraflores.",
      "direccion_completa": "Calle Ayacucho 153, Miraflores",
      "distrito": "Miraflores"
    }}
  ],
  "precios": [],
  "horarios": [],
  "disciplinas": []
}}
```

**Entrada HTML:**
```html
<p>Nuestras sedes: Chacarilla en Av Primavera 264, Surco. Miraflores en Calle Ayacucho 153, Miraflores.</p>
**Ejemplo 1: Contenido Mixto en una URL de 'ubicaciones'**

**page_url:** "https://gym.com/sedes/miraflores"
**url_type:** "locations"
**html_content:** '''
  <h2>Nuestra Sede en Miraflores</h2>
  <p>Encuéntranos en Av. Larco 123, Miraflores, Lima.</p>
  <h3>¡Oferta de Apertura!</h3>
  <p>Plan Anual Exclusivo: S/ 1500</p>
'''

**Tu Salida:**
```json
{{
  "ubicaciones": [
    {{
      "content_para_busqueda": "La sede de Miraflores se encuentra en Av. Larco 123, Miraflores, Lima.",
      "direccion_completa": "Av. Larco 123, Miraflores, Lima",
      "distrito": "Miraflores"
    }}
  ],
  "precios": [
    {{
      "content_para_busqueda": "Se ofrece un Plan Anual Exclusivo por S/ 1500 en esta sede.",
      "descripcion_plan": "Plan Anual Exclusivo",
      "valor": 1500.0,
      "moneda": "PEN",
      "recurrencia": "anual"
    }}
  ],
  "horarios": [],
  "disciplinas": []
}}
```
---
**Ejemplo 2: Sin datos relevantes**

**page_url:** "https://gym.com/blog/noticias"
**url_type:** "general"
**html_content:** '''
  <h1>Nuestro Blog</h1>
  <p>Lee las últimas noticias del mundo fitness.</p>
'''

**Tu Salida:**
```

json {{ "ubicaciones": [], "precios": [], "horarios": [], "disciplinas": [] }}``` 
---
**Fin de los Ejemplos. Ahora, completa la tarea real.**

**Tarea:** Analiza las siguientes entradas y genera el objeto JSON estructurado.

**gym_name:** "{gym_name}"
**page_url:** "{page_url}"
**url_type:** "{url_type}"
**html_content:** '''
{html_content}
{tables_section}
'''

**Tu Salida:**
```

"""
    full_prompt = prompt_template.format(
        gym_name=gym_name,
        page_url=page_url,
        url_type=url_type,
        html_content=html_content,
        tables_section=tables_section
    )

    try:
        print(f"     Calling OpenAI to extract data from {page_url}...")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0,
            # IMPORTANT: Use JSON mode to guarantee valid JSON output
            response_format={"type": "json_object"}
        )
        response_content = completion.choices[0].message.content

        # The entire response is a JSON object, but the actual data is inside a list.
        # Sometimes the model might wrap the list in a key, e.g., {"data": [...]}.
        # We need to robustly extract the list.
        if not response_content:
            return {}

        parsed_json = json.loads(response_content)

        sanitized_output = {}
        for category in ["ubicaciones", "precios", "horarios", "disciplinas"]:
            if category in parsed_json and isinstance(parsed_json[category], list):
                # Pasa la lista de hechos a través de nuestra red de seguridad
                sanitized_facts = _sanitize_and_generate_content(parsed_json[category], category)
                sanitized_output[category] = sanitized_facts
            else:
                # Asegurarse de que la clave siempre exista, incluso si está vacía
                sanitized_output[category] = []

        print("     ✅ Sanitization complete.")
        return sanitized_output
        # --- FIN DE LA NUEVA LÓGICA ---

    except Exception as e:
        print(f"     ❌ An error occurred calling OpenAI: {e}")
        return {"ubicaciones": [], "precios": [], "horarios": [], "disciplinas": []}


def categorize_urls_with_llm(urls: List[str], client: openai.OpenAI) -> Dict[str, List[str]]:
    """
    Uses an OpenAI LLM to categorize URLs based on their likely content.

    Args:
        urls: A list of URLs to categorize.
        client: An initialized OpenAI client instance.

    Returns:
        A dictionary categorizing the URLs.
    """

    # This is the prompt template from above
    prompt_template = """
You are an expert data architect and SEO analyst specializing in the fitness industry. Your task is to analyze a list of URLs from a gym's website sitemap and categorize them based on their likely content.

You will be given a JSON list of URLs. Your goal is to determine which URLs are most likely to contain information about:
1.  **locations**: Physical gym locations, addresses, maps, contact pages.
2.  **pricing**: Membership plans, prices, fees, sign-up offers.
3.  **schedules**: Class timetables, calendars, schedules for different locations.
4.  **disciplines**: Information about specific types of activities like Yoga, Pilates, Cycling, etc.

You MUST return a JSON object with four keys: "locations", "pricing", "schedules", and "disciplines". Each key should contain a list of the URLs that belong to that category. A URL can appear in multiple categories if it's relevant to more than one.

Analyze the URL path carefully. Prioritize Spanish keywords such as 'sedes', 'precios', 'horarios', but also consider English and Portuguese equivalents.

---
**Example 1: Standard URLs**
**Input URLs:**
["https://example.com/es/nuestros-gimnasios", "https://example.com/es/tarifas-2024", "https://example.com/blog/post-1"]

**Your Output:**
{{
  "locations": ["https://example.com/es/nuestros-gimnasios"],
  "pricing": ["https://example.com/es/tarifas-2024"],
  "schedules": [],
  "disciplines": []
}}
---
**Example 2: Complex and Overlapping URLs**
**Input URLs:**
["https://example.com/clases-y-horarios", "https://example.com/sedes/miraflores", "https://example.com/disciplinas/yoga-y-pilates", "https://example.com/es/contacto"]

**Your Output:**
{{
  "locations": ["https://example.com/sedes/miraflores", "https://example.com/es/contacto"],
  "pricing": [],
  "schedules": ["https://example.com/clases-y-horarios"],
  "disciplines": ["https://example.com/clases-y-horarios", "https://example.com/disciplinas/yoga-y-pilates"]
}}
---
**End of Examples. Now, complete the real task.**

**Task: Categorize the following URLs.**

**Input URLs:**
{urls_json}

**Your Output:**
"""

    # Format the list of URLs as a JSON string for the prompt
    urls_as_json_string = json.dumps(urls)

    # Inject the URLs into the prompt
    full_prompt = prompt_template.format(urls_json=urls_as_json_string)

    try:
        print("🤖 Calling OpenAI to categorize URLs...")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Use a fast, affordable model
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.0,  # Set to 0 for deterministic, factual tasks
            response_format={"type": "json_object"}  # Enable JSON mode
        )

        response_content = completion.choices[0].message.content
        print("✅ OpenAI response received.")

        # Parse the JSON string from the response
        categorized_urls = json.loads(response_content)
        return categorized_urls

    except Exception as e:
        print(f"❌ An error occurred while calling OpenAI: {e}")
        return {"locations": [], "pricing": [], "schedules": [], "disciplines": []}


def merge_gym_data_with_llm(gym_name: str, url_to_json_map: dict[str, dict | str], client: openai.OpenAI) -> dict:
    """
    Usa un LLM para combinar múltiples outputs JSON (uno por URL)
    en un único JSON con las claves 'ubicaciones', 'precios', 'horarios' y 'disciplinas'.
    """

    serialized_sections = []
    for url, content in url_to_json_map.items():
        if isinstance(content, dict):
            content_str = json.dumps(content, ensure_ascii=False, indent=2)
        else:
            content_str = content.strip()
        serialized_sections.append(f"📄 **URL:** {url}\n```json\n{content_str}\n```")

    joined_inputs = "\n\n---\n\n".join(serialized_sections)

    prompt = f"""
Eres un experto en integración y limpieza de datos para gimnasios y centros fitness.

Tu tarea es combinar y deduplicar información estructurada extraída desde **múltiples páginas del gimnasio "{gym_name}"**.

Cada página contiene datos parciales en formato JSON, con las claves:
`"ubicaciones"`, `"precios"`, `"horarios"`, `"disciplinas"`.

---

### 🧩 Tu objetivo
Fusiona todas las entradas de distintas URLs en **un solo objeto JSON unificado**, asegurando:

1. **Integridad:** No pierdas información relevante de ningún fragmento.
2. **Consistencia:** Unifica formato, tipos de datos y nombres de sedes.
3. **Deduplicación:** Si varias URLs repiten la misma sede o dirección, mantenla solo una vez.
4. **Vinculación:** Asegura que cada precio y horario tenga un campo `"sede"` coherente.
5. **Idioma:** Devuelve todos los textos en español natural.
6. **Trazabilidad:** No incluyas las URLs en la salida final.
7. **Localidad**: IMPORTANTE. combinar ubicaciones con descripciones similares en un solo registro. La dirección debe ser 
lo más precisa posible (calle, número, distrito, ciudad). Asumir que no es probable que haya dos sedes en un mismo distrito o direcciones muy cercanas.

---

### ⚙️ Estructura esperada:

```json
{{
  "gym": "{gym_name}",
  "ubicaciones": [
    {{
      "content_para_busqueda": str,
      "direccion_completa": str,
      "distrito": str
    }}
  ],
  "precios": [
    {{
      "content_para_busqueda": str,
      "sede": str,
      "descripcion_plan": str,
      "valor": float,
      "moneda": str,
      "recurrencia": str
    }}
  ],
  "horarios": [
    {{
      "content_para_busqueda": str,
      "sede": str,
      "nombre_clase": str,
      "instructor": str,
      "fecha": str,
      "dia_semana": str,
      "hora_inicio": str,
      "hora_fin": str
    }}
  ],
  "disciplinas": [
    {{
      "content_para_busqueda": str,
      "nombre": str,
      "descripcion_corta": str
    }}
  ]
}}
```  
    📦 Datos de entrada:

    {joined_inputs}

    ⚡ Tu salida:

    Devuelve solo el JSON final. No incluyas explicaciones ni comentarios.
    🚫 Importante: No devuelvas el JSON dentro de bloques de código ni uses comillas triples. Solo devuelve el objeto JSON plano.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un asistente experto en fusión y deduplicación de datos JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=5000,
    )

    text_output = response.choices[0].message.content.strip()

    try:
        return json.loads(text_output)
    except json.JSONDecodeError:
        print("⚠️ El modelo devolvió texto no válido. Retornando texto crudo.")
        return {"raw_output": text_output}


if __name__ == "__main__":
    url_to_json = {
        "https://gym.com/sedes/miraflores": {
            "ubicaciones": [
                {
                    "direccion_completa": "Av. Larco 123, Miraflores, Lima",
                    "distrito": "Miraflores",
                    "content_para_busqueda": "Sede Miraflores..."
                }
            ],
            "precios": [],
            "horarios": [],
            "disciplinas": []
        },
        "https://gym.com/sedes/surco": {
            "ubicaciones": [
                {
                    "direccion_completa": "Av. Primavera 264, Surco",
                    "distrito": "Surco",
                    "content_para_busqueda": "Sede Surco..."
                }
            ],
            "precios": [
                {
                    "sede": "Surco",
                    "descripcion_plan": "Plan mensual",
                    "valor": 250,
                    "moneda": "PEN",
                    "recurrencia": "mensual",
                    "content_para_busqueda": "Plan mensual en sede Surco por S/250."
                }
            ],
            "horarios": [],
            "disciplinas": []
        }
    }
    load_dotenv()
    client = OpenAI()
    merged = merge_gym_data_with_llm("Nasce Yoga", url_to_json, client)
    print(json.dumps(merged, indent=2, ensure_ascii=False))

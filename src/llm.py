import openai
import json
from typing import Dict, List, Dict, Any


def extract_structured_data(
        client: openai.OpenAI,
        page_url: str,
        url_type: str,
        html_content: str,
        gym_name: str
) -> List[Dict[str, Any]]:
    """
    Uses an OpenAI model to parse HTML and extract a list of structured "fact documents".
    """
    prompt_template = """
    You are a world-class data extraction agent for the fitness industry. Your sole purpose is to parse pruned HTML content from a gym's website and convert it into a structured JSON list of "fact documents".
**Your Goal:** Extract all relevant facts about the gym's **locations, pricing, schedules, and disciplines**. For each fact you find, you MUST generate a single JSON object containing both a structured `metadata` block and an unstructured `content` summary. The gym name is '{gym_name}'.
**Your Instructions:**
1. You will receive the `page_url`, the `url_type` (e.g., "pricing", "locations", "schedules", "homepage/general"), and the pruned `html_content`.
2. The `url_type` is a strong hint about the primary content on the page. If the type is "homepage/general", you must look for ALL types of information.
3. You MUST return a JSON list of fact objects.
4. For EACH fact object, you MUST generate a `content` field. This should be a single, concise English sentence summarizing the structured data in the `metadata`. This is essential for search.
5. You MUST populate the `metadata` with the structured data you extract. All fields are optional; only include what you can confidently find.
6. If after analyzing the HTML, you find NO relevant information, you MUST return an empty list `[]`. Do not invent or hallucinate data.

**Example 1: Input from a 'pricing' URL**
**page_url:** "[https://gym.com/planes](https://gym.com/planes)" **url_type:** "pricing" **html_content:** '''
### Plan Mensual
Acceso Total. S/ 180 por mes.
Compromiso 6 meses.
### Clase Suelta
S/ 45
'''
**Your Output:**``` json
[
  {{
    "gym_name": "{gym_name}",
    "source_url": "https://gym.com/planes",
    "content": "A monthly all-access plan is available for S/ 180 per month, with a 6-month commitment.",
    "metadata": {{
      "data_type": "pricing",
      "location_district": null,
      "price_value": 180.0,
      "price_currency": "PEN",
      "plan_type": "monthly",
      "plan_duration_months": 6
    }}
  }},
  {{
    "gym_name": "{gym_name}",
    "source_url": "https://gym.com/planes",
    "content": "A single drop-in class costs S/ 45.",
    "metadata": {{
      "data_type": "pricing",
      "location_district": null,
      "price_value": 45.0,
      "price_currency": "PEN",
      "plan_type": "per_class",
      "plan_duration_months": null
    }}
  }}
]
```

 
Example 2: Input from a 'schedules' URL
page_url: "https://siclo.com/pe/sedes/miraflores/horarios" url_type: "schedules" html_content: '''
Horarios para Miraflores
Lunes
Yoga
Ana
7:00 PM - 8:00 PM
'''
Your Output:``` json
[
  {{
    "gym_name": "Síclo",
    "source_url": "https://siclo.com/pe/sedes/miraflores/horarios",
    "content": "At the Miraflores location, Ana teaches a Yoga class on Mondays from 7:00 PM to 8:00 PM.",
    "metadata": {{
      "data_type": "schedule",
      "location_district": "Miraflores",
      "discipline": "Yoga",
      "day_of_week": "Monday",
      "start_time": "19:00",
      "end_time": "20:00",
      "instructor": "Ana"
    }}
  }}
]
```

 
Example 3: No relevant data found
page_url: "https://gym.com/blog/noticias" url_type: "general" html_content: '''
Nuestro Blog
Entérate de las últimas noticias del mundo fitness.
'''
Your Output:``` json
[]
```
**End of Examples. Now, complete the real task.**
**Task:** Analyze the following inputs and generate the list of fact documents.
**page_url:** "{page_url}" **url_type:** "{url_type}" **html_content:** ''' {html_content} '''
**Your Output:**
"""
    # Using .format() requires escaping the JSON braces with {{ and }}
    # But for the placeholder {html_content}, we use single braces.
    # The prompt is already formatted this way.

    full_prompt = prompt_template.format(
        gym_name=gym_name,
        page_url=page_url,
        url_type=url_type,
        html_content=html_content
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
            return []

        parsed_json = json.loads(response_content)

        if isinstance(parsed_json, dict):
            print(f"     ✅ Extracted single fact.")
            return [parsed_json]

        # If the top-level object is the list itself (less common but possible)
        if isinstance(parsed_json, list):
            print(f"     ✅ Extracted {len(parsed_json)} facts.")
            return parsed_json

        return []  # Return empty if no list is found

    except Exception as e:
        print(f"     ❌ An error occurred calling OpenAI: {e}")
        return []


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

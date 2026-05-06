import json
import os
import random
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE  = "data/lek_pl_sample.json"
OUTPUT_FILE = "data/medical_keywords.json"
OUTPUT_TXT  = "data/medical_keywords.txt"

MODEL      = "gpt-4o-mini"
BATCH_SIZE = 20
SLEEP_SEC  = 1.0

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """Jesteś ekspertem medycznym. Twoim zadaniem jest wydobycie pojęć medycznych
z pytań egzaminacyjnych LEK/LDEK (Lekarski Egzamin Końcowy).

Z podanego tekstu wyodrębnij WSZYSTKIE pojęcia, które warto mieć w bazie wiedzy RAG:
- nazwy chorób i zespołów chorobowych (np. "zawał mięśnia sercowego", "zespół Cushinga")
- leki i substancje czynne (np. "metoprolol", "amoksycylina")
- procedury i zabiegi medyczne (np. "koronarografia", "laparoskopia")
- badania diagnostyczne (np. "EKG", "troponina", "kolonoskopia")
- struktury anatomiczne istotne klinicznie (np. "zastawka mitralna", "nerw błędny")
- objawy i znaki kliniczne (np. "duszność", "ból w klatce piersiowej")
- wartości i parametry laboratoryjne (np. "INR", "eGFR", "TSH")
- klasyfikacje i skale kliniczne (np. "NYHA III", "FIGO I")
- specjalności i konteksty kliniczne (np. "kardiologia", "onkologia ginekologiczna")

Zwróć TYLKO listę JSON z unikalnymi pojęciami (strings), bez duplikatów, bez komentarzy.
Format: ["pojęcie1", "pojęcie2", ...]

Nie dodawaj żadnego tekstu poza JSON."""


def load_questions(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = list(data.get("question_w_options", {}).values())
    print(f"Loaded {len(questions)} questions.")
    return questions


def extract_keywords_batch(questions: list[str]) -> list[str]:
    combined = "\n\n---\n\n".join(
        f"[{i+1}] {q}" for i, q in enumerate(questions)
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": combined},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        keywords: list[str] = json.loads(raw)
        return [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
    except Exception as e:
        print(f"Batch parse error: {e}")
        return []


def shuffle_avoiding_neighbors(items: list[str]) -> list[str]:
    items = list(items)
    random.shuffle(items)

    def similar(a: str, b: str, n: int = 8) -> bool:
        return a[:n].lower() == b[:n].lower()

    for _ in range(5):
        improved = False
        for i in range(1, len(items)):
            if similar(items[i - 1], items[i]):
                for j in range(i + 1, min(i + 30, len(items))):
                    if not similar(items[i - 1], items[j]) and not similar(items[j], items[i]):
                        items[i], items[j] = items[j], items[i]
                        improved = True
                        break
        if not improved:
            break

    return items


def save_results(keywords: list[str], json_path: str, txt_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(keywords))
    print(f"Saved {len(keywords)} keywords to {json_path} and {txt_path}")


def main() -> None:
    questions = load_questions(INPUT_FILE)

    all_keywords: set[str] = set()
    total_batches = (len(questions) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(questions))
        batch = questions[start:end]

        keywords = extract_keywords_batch(batch)
        new = len(keywords) - len(all_keywords.intersection(keywords))
        all_keywords.update(keywords)
        print(f"Batch {batch_idx + 1}/{total_batches} done, {new} new keywords (total unique: {len(all_keywords)})")

        if batch_idx < total_batches - 1:
            time.sleep(SLEEP_SEC)

    shuffled = shuffle_avoiding_neighbors(list(all_keywords))
    save_results(shuffled, OUTPUT_FILE, OUTPUT_TXT)

    print("\nSample (first 30):")
    for kw in shuffled[:30]:
        print(f"  {kw}")


if __name__ == "__main__":
    main()
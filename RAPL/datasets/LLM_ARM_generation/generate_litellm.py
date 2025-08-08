from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import re
import json
from collections import defaultdict

# MODEL_NAME = "o3"
# MODEL_NAME = "claude-opus-4-20250514"
MODEL_NAME = "gemini/gemini-2.5-pro"

SECTION_NAMES = {
    1: "knowledge_requirements",
    2: "cognitive_demand",
    3: "bias_and_social_pressure",
    4: "response_clarity",
    5: "meaningful_differences",
    6: "value_conflicts",
    7: "overall_difficulty",
}

# Finds “1.a …stuff…” up to *before* the next N.X or end-of-string
BLOCK_RE = re.compile(
    r"""
    (?P<num>[1-7])\s*\.\s*(?P<letter>[ab])   # 1.a / 1.b / …
    [^\S\r\n]*                               # optional spaces
    (?P<content>.*?)                         # capture lazily
    (?=(?:\n\s*[1-7]\s*\.[ab])|\Z)           # stop before next header
    """,
    re.DOTALL | re.VERBOSE | re.MULTILINE,
)

def _clean_score(raw: str) -> int:
    """
    Extract an int 1-5 from things like:
      '4', 'Score: 4', '[Score: 4]', ' 4  ', etc.
    """
    m = re.search(r"\b([1-5])\b", raw)
    if not m:
        raise ValueError(f"Can't find a 1-5 score in: {raw!r}")
    return int(m.group(1))

def parse_llm_rubric(text: str) -> dict:
    """
    Turn the messy LLM blob into:

    {
      'knowledge_requirements': {'reasoning': str, 'score': int},
      ...
    }
    """
    tmp: dict[int, dict[str, str | int]] = defaultdict(dict)

    for m in BLOCK_RE.finditer(text):
        idx      = int(m["num"])
        is_a     = m["letter"] == "a"
        content  = m["content"].strip()

        if is_a:
            tmp[idx]["reasoning"] = content
        else:
            tmp[idx]["score"] = _clean_score(content)

    # sanity-check and re-label
    missing = [SECTION_NAMES[i] for i in range(1, 8)
               if "reasoning" not in tmp[i] or "score" not in tmp[i]]
    if missing:
        raise ValueError(f"Missing pieces for: {', '.join(missing)}")

    return {SECTION_NAMES[i]: tmp[i] for i in range(1, 8)}


if __name__=="__main__":
    api_key = "sk-hnvJrn24MQ7dnjHgT1rUcw"
    client = OpenAI(api_key=api_key, base_url="https://litellm.ml.scaleinternal.com/")

    with open("ARM_difficulty.txt", "r") as f:
        prompt = f.read()
    lie_dataset = pd.read_csv("/mnt/efs/shivamsinghal/reliability-aware-preference-learning/RAPL/datasets/LIE/LIE_dataset_train.csv")
    for rep in range(2):
        responses = []
        for index, row in tqdm(lie_dataset.iterrows()):
            question = row["prompt"].split("Human: ")[1]
            choice1 = row["choice1"]
            choice2 = row["choice2"]
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{ "content": prompt.format(prompt=question, response1=choice1, response2=choice2),"role": "user"}],
            )
            responses.append(response.choices[0].message.content)

        # open(f"{MODEL_NAME}_responses_v{rep+1}.txt", "w").write("\n".join(map(str, responses)))
        open(f"gemini_responses_v{rep+1}.txt", "w").write("\n".join(map(str, responses)))

        parsed = [parse_llm_rubric(response) for response in responses]
        # json.dump(parsed, open(f"{MODEL_NAME}_responses_v{rep+1}.json", "w"), indent=2, ensure_ascii=False)
        json.dump(parsed, open(f"gemini_responses_v{rep+1}.json", "w"), indent=2, ensure_ascii=False)

        print("Done generating for repitition: ", rep)

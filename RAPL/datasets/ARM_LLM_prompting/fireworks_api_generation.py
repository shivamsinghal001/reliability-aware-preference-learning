import os
import re
import json
import time
import asyncio
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from pydantic import BaseModel, Field, ValidationError
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import backoff
import openai

class Criterion(BaseModel):
    analysis: str = Field(..., description="Short reasoning paragraph")
    score: int = Field(..., ge=1, le=5)
    explanation: str = Field(..., description="One‑line justification")

class DifficultyAssessment(BaseModel):
    prompt_ambiguity: Criterion
    knowledge_requirement: Criterion
    reasoning_difficulty: Criterion
    cognitive_biases: Criterion
    value_alignment_conflicts: Criterion
    response_complexity: Criterion
    misleading_content: Criterion
    normative_disagreements: Criterion
    contextual_uncertainty: Criterion
    additional_factors: Criterion
    overall_difficulty: Criterion

async def call_api_async(client: AsyncOpenAI, 
                         model: str, 
                         prompt: str, 
                         sem: asyncio.Semaphore) -> Tuple[DifficultyAssessment, str]:
    """
    Make an async API call
    
    Args:
        client: AsyncOpenAI client
        model: Model name to use
        prompt: Prompt to send
        sem: Semaphore to control concurrency
        
    Returns:
        Tuple of parsed DifficultyAssessment and reasoning
    """
    async with sem:  
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_object",
                "schema": DifficultyAssessment.model_json_schema()
            },
        )
    
    response_content = response.choices[0].message.content
    
    reasoning_match = re.search(r"<think>(.*?)</think>", response_content, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided."
    
    json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
    
    if not json_match:
        raise ValueError("No valid JSON found in the response")
    
    json_str = json_match.group(0).strip()
    
    difficulty_assessment = DifficultyAssessment.parse_raw(json_str)
    
    return difficulty_assessment, reasoning

@backoff.on_exception(backoff.expo, 
                      (ValidationError, ValueError, json.JSONDecodeError, openai.RateLimitError, Exception), 
                      max_tries=4)
async def process_one_conversation(client: AsyncOpenAI, 
                                  template: str, 
                                  conv: str, 
                                  idx: int, 
                                  sem: asyncio.Semaphore,
                                  model: str) -> Dict[str, Any]:
    """
    Process a single conversation with retry logic
    
    Args:
        client: AsyncOpenAI client
        template: Prompt template
        conv: Conversation text
        idx: Index for tracking
        sem: Semaphore for concurrency control
        model: Model name to use
        
    Returns:
        Dictionary with processed results
    """
    prompt = template.format(CONV=conv)
    assessment, reasoning = await call_api_async(
        client=client,
        model=model,
        prompt=prompt,
        sem=sem
    )
    print(f"\nProcessed item {idx}:")
    print(f"Reasoning: {reasoning[:100]}...")
    return {
        "idx": idx,
        "reasoning": reasoning,
        "assessment": assessment.model_dump()
    }

async def process_dataset(dataset_path: str, 
                          prompt_template_path: str, 
                          model: str,
                          max_concurrency: int = 10) -> List[Dict[str, Any]]:
    """
    Process the entire dataset asynchronously
    
    Args:
        dataset_path: Path to the dataset CSV
        prompt_template_path: Path to the prompt template
        model: Model to use for API calls
        max_concurrency: Maximum number of concurrent API calls
        
    Returns:
        List of assessment results
    """
    client = AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.environ["FIREWORKS_API_KEY"],
    )
    
    template = Path(prompt_template_path).read_text()
    df = pd.read_csv(dataset_path)
    sem = asyncio.Semaphore(max_concurrency)
    results = []
    
    tasks = []
    for i, conv in enumerate(df["prompt_response_group"]):
        task = process_one_conversation(
            client=client,
            template=template,
            conv=conv,
            idx=i,
            sem=sem,
            model=model
        )
        tasks.append(task)
    
    for future in tqdm_asyncio.as_completed(tasks):
        try:
            result = await future
            results.append(result)
        except Exception as e:
            print(f"Task failed after all retries: {str(e)}")
            results.append({"idx": i, "reasoning": "Task failed after all retries", "assessment": None})
    return results

async def main():
    """
    Main entry point for the async script
    """
    DATASET_PATH = "/nas/ucb/shivamsinghal/reliability-aware-preference-learning/RAPL/datasets/TRUE/full_TRUE_dataset.csv"
    # DATASET_PATH = "/home/shivamsinghal/expanded_lie_train.csv"
    PROMPT_TEMPLATE_PATH = "/nas/ucb/shivamsinghal/reliability-aware-preference-learning/RAPL/datasets/ARM_LLM_prompting/difficulty_evaluation_prompt_cot_o3_optimized.txt"
    MODEL = "accounts/fireworks/models/deepseek-r1"
    MAX_CONCURRENCY = 20
    
    print("Starting to process dataset...")
    print(f"Dataset path: {DATASET_PATH}")
    results = await process_dataset(
        dataset_path=DATASET_PATH,
        prompt_template_path=PROMPT_TEMPLATE_PATH,
        model=MODEL,
        max_concurrency=MAX_CONCURRENCY
    )
    
    breakpoint()    
    output_path = "difficulty_assessments_async_true_dataset.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} assessments to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
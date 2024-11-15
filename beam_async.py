import openai
import os
import anthropic
import asyncio
import argparse
import json
from typing import List, Tuple, Optional
import sys
from dataclasses import dataclass
from copy import deepcopy
import numpy as np

GENERATOR_SYSTEM_PROMPT = """You are an assistant that solves math problems step by step. 
Each step should be about a single high-level action. You end a step with a newline. 
If you come up with an answer, please provide it with The answer is:"""

GENERATOR_PROMPT_TEMPLATE = """Question: {question}
Solution Steps:
{solution_steps}
Next step:"""

GENERATE_MODEL = "gpt-4o-mini"
REWARD_MODEL = "/mnt/chatbot30TB/junghyunlee/juneyang/Qwen/Qwen2.5-Math-RM-72B"
TERMINATION_KEYWORDS = ['Answer', 'answer']

@dataclass
class SingleBeam:
    steps: List[str]
    scores: List[float]
    ended: bool = False
    answer_location: Optional[int] = None

    @property
    def total_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

class BeamManager:
    def __init__(self, question: str, beam_width: int):
        self.question = question
        self.beam_width = beam_width
        self.beams: List[SingleBeam] = [SingleBeam(steps=[], scores=[])]

    def expand_beams(self, new_steps_per_beam: List[List[str]], new_scores_per_beam: List[List[float]]) -> None:

        new_beams: List[SingleBeam] = []
        
        # Expand each existing beam
        for beam_idx, beam in enumerate(self.beams):
            new_steps = new_steps_per_beam[beam_idx]
            new_scores = new_scores_per_beam[beam_idx]
            
            # Create new beams for each expansion
            for step, score in zip(new_steps, new_scores):
                new_beam = SingleBeam(
                    steps=beam.steps + [step],
                    scores=beam.scores + [score],
                    ended=any(keyword in step.lower() for keyword in TERMINATION_KEYWORDS)
                )
                if new_beam.ended:
                    new_beam.answer_location = len(new_beam.steps) - 1
                new_beams.append(new_beam)

        # Sort and prune
        new_beams.sort(key=lambda x: x.total_score, reverse=True)
        self.beams = new_beams[:self.beam_width]

    def all_beams_ended(self) -> bool:
        return all(beam.ended for beam in self.beams)

    def get_current_steps(self) -> List[List[str]]:
        return [beam.steps for beam in self.beams]

class BeamSearcher:
    def __init__(self, generator_client, anthropic_client, rm_client):
        self.generator_client = generator_client
        self.anthropic_client = anthropic_client
        self.rm_client = rm_client

    async def generate_steps_batch(self, question: str, beam_steps: List[str], expansion: int) -> List[str]:
        prompt = GENERATOR_PROMPT_TEMPLATE.format(
            question=question,
            solution_steps="\n".join(beam_steps)
        )

        try:
            response = await asyncio.to_thread(
                self.generator_client.chat.completions.create,
                model=GENERATE_MODEL,
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                n=expansion,
                stop=["\n"],
                temperature=0.7
            )
            return [choice.message.content for choice in response.choices]
        except Exception as e:
            print(f"Error during batch generation: {e}")
            return [""] * expansion

    async def generate_all_steps(self, question: str, all_beam_steps: List[List[str]], expansion: int) -> List[List[str]]:
        tasks = [
            self.generate_steps_batch(question, beam_steps, expansion)
            for beam_steps in all_beam_steps
        ]
        return await asyncio.gather(*tasks)

    async def verify_all_steps_rm(
        self, 
        question: str, 
        all_beam_steps: List[List[str]], 
        all_new_steps: List[List[str]]
    ) -> List[List[float]]:
        all_scores = []
        
        verification_texts = []
        step_indices = []
        
        for beam_idx, (beam_steps, new_steps) in enumerate(zip(all_beam_steps, all_new_steps)):
            context = question + "\n" + "\n".join(beam_steps)
            for step_idx, new_step in enumerate(new_steps):
                verification_text = f"{context}\n{new_step}"
                verification_texts.append(verification_text)
                step_indices.append((beam_idx, step_idx))
        
        try:
            print("Calling RM for verification...")
            response = await asyncio.to_thread(
                self.rm_client.embeddings.create,
                model=REWARD_MODEL,
                input=verification_texts
            )
            
            scores_by_beam = [[] for _ in all_beam_steps]
            
            for idx, embedding_data in enumerate(response.data):
                beam_idx, step_idx = step_indices[idx]
                token_scores = embedding_data.embedding
                
                step_score = token_scores[-1]
                
                normalized_score = max(0.0, min(1.0, step_score))
                
                while len(scores_by_beam[beam_idx]) <= step_idx:
                    scores_by_beam[beam_idx].append(0.0)
                scores_by_beam[beam_idx][step_idx] = normalized_score
            
            return scores_by_beam
            
        except Exception as e:
            print(f"Error during reward model verification: {e}")
            return [[0.0] * len(new_steps) for new_steps in all_new_steps]

    async def beam_search(
        self, 
        question: str, 
        total_generations: int, 
        beam_width: int, 
        max_steps: int
    ) -> List[SingleBeam]:
        """Perform beam search with parallel generation and reward model verification."""
        beam_manager = BeamManager(question, beam_width)
        expansions_per_beam = max(1, total_generations // beam_width)
        
        for step_num in range(1, max_steps + 1):
            print(f"\n--- Step {step_num} ---")
            
            current_beam_steps = beam_manager.get_current_steps()
            
            all_new_steps = await self.generate_all_steps(
                question, 
                current_beam_steps,
                expansions_per_beam
            )
            
            all_new_scores = await self.verify_all_steps_rm(
                question,
                current_beam_steps,
                all_new_steps
            )
            
            beam_manager.expand_beams(all_new_steps, all_new_scores)
            
            print(f"\nTop {beam_width} beams after Step {step_num}:")
            for idx, beam in enumerate(beam_manager.beams, 1):
                print(f"Beam {idx}: Score={beam.total_score:.4f}")
                steps = '\n'.join(beam.steps)
                print(f"Steps:\n{steps}\n")
            
            if beam_manager.all_beams_ended():
                print("All beams have ended. Terminating search.")
                break
                
        return beam_manager.beams

    def search(self, question: str, total_generations: int,
               beam_width: int, max_steps: int) -> List[SingleBeam]:
        return asyncio.run(self.beam_search(question,
                                            total_generations,
                                            beam_width,
                                            max_steps))

def main():
    parser = argparse.ArgumentParser(description='Beam Search with Verifiers')
    parser.add_argument('--question', nargs='*', help='The math problem to solve')
    parser.add_argument('--total-generations', type=int, default=8,
                      help='Total number of parallel generations')
    parser.add_argument('--beam-width', type=int, default=2,
                      help='Number of top beams to keep')
    parser.add_argument('--max-steps', type=int, default=20,
                      help='Maximum number of steps')
    parser.add_argument('--config', type=str, default='api_keys.json',
                      help='Path to API keys configuration file')
    parser.add_argument('--generator', type=str, default='gpt-4o-mini',
                      help='Generator LLM model')
    args = parser.parse_args()

    try:
        with open(args.config) as config_file:
            api_keys = json.load(config_file)
            os.environ["OPENAI_API_KEY"] = api_keys["openai"]
            os.environ["ANTHROPIC_API_KEY"] = api_keys["anthropic"]
    except FileNotFoundError:
        print(f"Error: Could not find config file: {args.config}")
        sys.exit(1)

    generator_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    rm_client = openai.OpenAI(api_key="dummy-key", base_url="http://0.0.0.0:8001/v1")

    question = ' '.join(args.question) if args.question else input("Enter the math problem you want to solve: ")
    print(f"\nSolving the problem: {question}")

    beam_searcher = BeamSearcher(generator_client, anthropic_client, rm_client)
    final_beams = beam_searcher.search(
        question,
        args.total_generations,
        args.beam_width,
        args.max_steps
    )

    print("\n--- Final Solutions ---")
    for idx, beam in enumerate(final_beams, 1):
        print(f"\nSolution {idx} (Average Score: {beam.total_score:.4f}):")
        for step_num, step in enumerate(beam.steps, 1):
            print(f"Step {step_num}: {step}")

if __name__ == "__main__":
    main()
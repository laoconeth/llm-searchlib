import openai
import os
from typing import List, Tuple
import sys
import json
import anthropic
import argparse


with open('api_keys.json') as config_file:
    api_keys = json.load(config_file)


os.environ["OPENAI_API_KEY"] = api_keys["openai"]

os.environ["ANTHROPIC_API_KEY"] = api_keys["anthropic"]



anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)


client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

TOTAL_GENERATIONS = 8  # Total number of parallel generations (i)
BEAM_WIDTH = 2          # Number of top beams to keep (k)
MAX_STEPS = 20          # Maximum number of steps to prevent infinite loops

GENERATE_MODEL = "gpt-4o-mini"
VERIFY_MODEL = "gpt-4o-mini"


class Beam:
    def __init__(self, question: str, steps: List[str], score: List[float]):
        """
        Initialize a Beam with the current steps and cumulative score.
        """
        self.question = question
        self.steps = steps
        self.score = score
        self.ended = False
        self.answer_location = None

    def __repr__(self):
        return f"Beam(steps={"\n".join(self.steps)}\n\nScore={self.score})"



def generate_step(question: str, steps: List[str], expansion: int = 2) -> str:
    """
    Generate the next step of the solution using OpenAI's completion API.
    Stops generation at the first newline.
    """
    prompt = f"Question: {question}\nSolution Steps:\n"
    solution_steps = "\n".join(steps)

    try:
        response = client.chat.completions.create(
            model=GENERATE_MODEL,
            messages=[
                {"role": "system", "content": "You are an assistant that solves math problems step by step. Each step should be about a single high-level action. You end a step with a newline. If you come up with an answer, please provide it with The answer is:"},
                # {"role": "system", "content": "You are an assistant that solves math problems step by step."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": solution_steps}
            ],
            max_tokens=150,
            n=expansion,
            stop=["\n"],
            temperature=0.7  # Set to 0 for deterministic output
        )


        # message = anthropic_client.messages.create(
        #     # model="claude-3-5-sonnet-20240620",
        #     model="claude-3-haiku-20240307",
        #     messages=[
        #         {"role": "system", "content": "You are an assistant that solves math problems step by step. Each step should be about a single high-level action. You end a step with a newline. If you come up with an answer, please provide it with The answer is:"},
        #         # {"role": "system", "content": "You are an assistant that solves math problems step by step."},
        #         {"role": "user", "content": prompt},
        #         {"role": "assistant", "content": solution_steps}
        #     ],
        #     max_tokens=150,
        #     n=expansion,
        #     stop=["\n"],
        #     temperature=0.7  # Set to 0 for deterministic output
        # )


        generated_step = [response.choices[idx].message.content for idx in range(expansion)]
        return generated_step
    except Exception as e:
        print(f"Error during generation: {e}")
        sys.exit(1)



def verify_step(question: str, prev_steps: List[str], new_step_candidates: List[str]) -> float:

    score_per_candidate = []

    for solution in new_step_candidates:
        prompt = f"Question: {question}\n\nPrevious solution steps:\n\n{'\n'.join(prev_steps)}\n\nCurrent step:\n\n{solution}\n\n"
        prompt += "Please directly provide a score between 0.0 and 1.0 for the correctness and relevance of the latest solution step without any additional text."

        try:
            response = client.chat.completions.create(
                model=VERIFY_MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant that evaluates the correctness of solution steps."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                n=1,
                stop=None,
                temperature=0.0  # Set to 0 for deterministic output
            )
            # Extract the score from the response
            score_text = response.choices[0].message.content
            # Attempt to parse the score
            score = float(score_text)
            # Ensure the score is between 0 and 1
            score = max(0.0, min(score, 1.0))
            score_per_candidate.append(score)
        except Exception as e:
            print(f"Error during verification: {e}")
            # In case of error, assign a default low score
            score_per_candidate.append(0.0)
        except ValueError:
            print(f"Invalid score format received: '{score_text}'. Assigning score 0.0.")
            score_per_candidate.append(0.0)
        
    return score_per_candidate





def beam_search(question: str, total_generations: int, beam_width: int, max_steps: int) -> List[Beam]:
    """
    Perform beam search to solve the given math problem.
    """
    beams = [Beam(question=question, steps=[], score=[]) for _ in range(beam_width)]

    for step_num in range(1, max_steps + 1):
        all_candidates = []

        print(f"\n--- Step {step_num} ---")
        print(f"Current number of beams: {len(beams)}")

        # Determine how many expansions per beam
        expansions_per_beam = total_generations // beam_width
        if expansions_per_beam == 0:
            expansions_per_beam = 1

        for beam in beams:
            new_candidate_steps = generate_step(question, beam.steps, expansion=expansions_per_beam) # returns a list of candidate steps
            scores = verify_step(question, beam.steps, new_candidate_steps) # Verify and score new candidate steps
            # Create a new candidate beams with the added step and updated score
            for idx, (new_step, score) in enumerate(zip(new_candidate_steps, scores)):

                new_beam_steps = beam.steps + [new_step]
                new_beam_score = beam.score + [score]  # Cumulative scoring
                new_beam = Beam(question=question, steps=new_beam_steps, score=new_beam_score)
                all_candidates.append(new_beam)
                print(f"Generated Step: '{new_step}' with score {score:.4f}")

        # If no candidates were generated, terminate
        if not all_candidates:
            print("No candidates generated. Terminating beam search.")
            break

        # Sort all candidates by their cumulative score in descending order
        all_candidates.sort(key=lambda b: sum(b.score), reverse=True)

        beams = all_candidates[:beam_width]
        print(f"Top {beam_width} beams after Step {step_num}:")
        for idx, beam in enumerate(beams, 1):
            print(f"Beam {idx}: Score={beam.score[-1]:.4f},\n\nSteps={'\n\n'.join(beam.steps)}\n\n")


        termination_keywords = ['Answer', 'answer']

        for beam in beams:
            if any(keyword in beam.steps[-1].lower() for keyword in termination_keywords):
                beam.ended = True
                beam.answer_location = len(beam.steps) - 1

        if all(beam.ended for beam in beams):
            print("Termination condition met. Ending beam search.")
            break


    return beams

def main():

    parser = argparse.ArgumentParser(description="Beam search solver for math problems.")
    parser.add_argument('question', type=str, nargs='*', help='The math problem you want to solve')
    args = parser.parse_args()

    # Get the math problem from the user
    if not args.question:
        question = input("Enter the math problem you want to solve: ")
    else:
        question = ' '.join(args.question)

    print(f"\nSolving the problem: {question}")

    # Perform beam search
    final_beams = beam_search(question, TOTAL_GENERATIONS, BEAM_WIDTH, MAX_STEPS)

    # Display the final solutions
    print("\n--- Final Solutions ---")
    for idx, beam in enumerate(final_beams, 1):
        for step_num, step in enumerate(beam.steps, 1):
            print(f"Step {step_num}: {step}")

if __name__ == "__main__":
    main()

import argparse
import subprocess
import sys

def run_step(step_number):
    scripts = {
        1: "01_langsmith_rag_pipeline.py",
        2: "02_prompt_hub_ab_routing.py",
        3: "03_ragas_evaluation.py",
        4: "04_guardrails_validator.py"
    }
    
    script = scripts.get(step_number)
    if not script:
        print(f"❌ Invalid step number: {step_number}")
        return

    print(f"\n\n{'#'*60}")
    print(f"### RUNNING STEP {step_number}: {script}")
    print(f"{'#'*60}\n")
    
    try:
        # Using subprocess.run to execute the scripts
        # We use sys.executable to ensure we use the same python interpreter
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step {step_number} failed with error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Day 22 Lab Steps")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Run a specific step (1-4)")
    
    args = parser.parse_args()
    
    if args.step:
        run_step(args.step)
    else:
        print("🚀 Starting full lab execution...")
        for i in range(1, 5):
            run_step(i)
        print("\n\n✅ ALL STEPS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()

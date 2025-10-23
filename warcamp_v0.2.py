import ollama
import chromadb
import time
import json
import os
import re

# --- CONFIGURATION ---
# The specific models for each role
SARGE_MODEL = 'codegemma:instruct' # 7b instruct model
ORCH_MODEL = 'codegemma:2b'       # 2b code-completion model

# This is where your AI will build its project
OUTPUT_DIRECTORY = "output_project"

# --- The "Council's Memory" (Task Queue) ---
# We use ChromaDB as a persistent task queue.
print("Initializing Council memory banks (ChromaDB)...")
# This will create a 'warcamp_db' folder inside your 'Warcamp' folder
client = chromadb.PersistentClient(path="./warcamp_db") 
council_tasks = client.get_or_create_collection(name="task_queue")
print("...Council is in session.")

# --- AGENT DEFINITIONS (v0.2) ---

class Sarge:
    """
    Sarge (CodeGemma Instruct).
    Receives the high-level goal and creates a detailed, step-by-step
    task list for the Orchs.
    """
    def __init__(self, model_name=SARGE_MODEL):
        self.model_name = model_name
        print(f"Sarge on deck. (Using {self.model_name})")

    def create_task_list(self, goal):
        print(f"\nSarge: 'ALRIGHT, YOU GRUNTS! New goal from the Chief: \"{goal}\"'")
        print("Sarge: 'I'm breaking it down into tasks... Work Work.'")
        print("(This may take a moment, Sarge is thinking...)")

        system_prompt = """
        You are "Sarge", a senior software architect.
        Your job is to take a high-level goal and break it down into a JSON list
        of specific, granular coding tasks.
        
        The tasks should be in logical order.
        Each task must be a single, clear instruction for a junior developer (an "Orch").
        Focus on creating files, writing code to them, and installing dependencies.
        
        Example:
        Goal: "make a simple python flask web server"
        Response:
        {
          "tasks": [
            "Create a new directory named 'output_project' for the code.",
            "Create a file named 'requirements.txt' inside 'output_project'.",
            "Add the line 'flask' to 'output_project/requirements.txt'.",
            "Create a file named 'app.py' inside 'output_project'.",
            "Write the python code for a basic flask server in 'output_project/app.py' that runs on port 5000 and has a single '/' route that returns 'Work complete!'."
          ]
        }
        
        ONLY output the JSON object. No other text or explanation.
        """
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Goal: \"{goal}\""}
                ],
                format='json'  # Request JSON output
            )
            
            task_data = json.loads(response['message']['content'])
            tasks = task_data.get('tasks', [])
            
            if not tasks:
                print("Sarge: 'Failed to create tasks. The response was empty!'")
                return []
                
            print(f"Sarge: 'I have {len(tasks)} tasks. Ready for the Council.'")
            return tasks
            
        except Exception as e:
            print(f"Sarge: 'My plan failed! Could not talk to Ollama.'")
            print(f"Error: {e}")
            return []

class Council:
    """
    The Council (ChromaDB).
    Manages the task queue.
    """
    def __init__(self, task_collection):
        self.tasks = task_collection
        # Clear old tasks from previous runs
        self.tasks.delete(where={"status": {"$in": ["pending", "complete"]}})
        print("Council: 'The old scrolls are burned. Ready for new tasks.'")

    def store_tasks(self, task_list):
        if not task_list:
            print("Council: 'No tasks to store.'")
            return
            
        print(f"Council: 'Recording {len(task_list)} new tasks in the archives.'")
        for i, task_desc in enumerate(task_list):
            self.tasks.add(
                documents=[task_desc],
                metadatas=[{"status": "pending", "task_num": i}],
                ids=[f"task_{i}"]
            )

    def get_next_task(self):
        """Gets the next available 'pending' task."""
        next_task = self.tasks.get(
            where={"status": "pending"},
            limit=1,
            include=["metadatas", "documents"]
        )
        
        if not next_task['ids']:
            print("Council: 'All tasks are complete.'")
            return None
            
        return {
            "id": next_task['ids'][0],
            "description": next_task['documents'][0]
        }
        
    def mark_task_complete(self, task_id):
        """Updates a task's status to 'complete'."""
        self.tasks.update(
            ids=[task_id],
            metadatas=[{"status": "complete"}]
        )
        # We don't print here, it's too noisy
        # print(f"Council: 'Task {task_id} has been marked complete.'")

class Orch:
    """
    The Orch (CodeGemma 2b).
    The worker. Takes a single task and writes the code.
    """
    def __init__(self, model_name=ORCH_MODEL, base_dir=OUTPUT_DIRECTORY):
        self.model_name = model_name
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        print(f"Orch ready for work. (Using {self.model_name})")

    def execute_task(self, task):
        print(f"\nOrch: 'Work work. New task: \"{task['description']}\"'")
        
        # Give the Orch context about the project structure
        project_context = self.get_project_context()
        
        system_prompt = f"""
        You are "Orch", a junior coder.
        Your job is to execute the given task and write the code.
        
        You must write *ONLY* the raw code, or a simple "Done." message for non-code tasks.
        Do not add explanations, markdown like ```python, or introductions.
        
        Current Project Structure:
        {project_context}
        
        Your output will be written to a file, so it must be clean.
        
        Example for a file-writing task:
        Task: "Write the python code for a basic flask server in 'output_project/app.py'"
        Your Response:
        from flask import Flask
        app = Flask(__name__)
        
        @app.route('/')
        def home():
            return 'Work complete!'
        
        if __name__ == '__main__':
            app.run(debug=True, port=5000)
        
        Example for a non-code task:
        Task: "Create a new directory named 'output_project'"
        Your Response:
        Done.
        """
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Task: \"{task['description']}\""}
                ]
            )
            
            output = response['message']['content']
            
            # --- FILE SYSTEM LOGIC ---
            self.process_orch_output(task['description'], output)
            
            print(f"Orch: 'Task done.'")
            return True
            
        except Exception as e:
            print(f"Orch: 'Work failed!'")
            print(f"Error: {e}")
            return False

    def get_project_context(self):
        """Scans the output directory to give context to the AI."""
        context = f"Base Directory: {self.base_dir}/\n"
        for root, dirs, files in os.walk(self.base_dir):
            rel_root = os.path.relpath(root, self.base_dir)
            if rel_root == '.':
                rel_root = ''
                
            for name in files:
                context += f"- {os.path.join(rel_root, name)}\n"
            for name in dirs:
                context += f"- {os.path.join(rel_root, name)}/\n"
        return context

    def process_orch_output(self, task_desc, output):
        """
        A simple helper to decide what to do with the AI's output.
        """
        # Clean up common markdown fences
        output = re.sub(r"```[\w\n]*", "", output).strip()

        # Try to find a filepath in the task description
        match = re.search(r"'(.*?)'", task_desc)
        if match:
            filepath_raw = match.group(1)
            
            # Prevent the AI from trying to write to itself
            if filepath_raw == 'warcamp_v0.2.py' or filepath_raw == 'requirements.txt':
                 print(f"Orch: 'Me no overwrite Warcamp scrolls!'")
                 return
                 
            # Ensure the path is safe and inside the base directory
            abs_path = os.path.abspath(os.path.join(self.base_dir, filepath_raw))
            if not abs_path.startswith(os.path.abspath(self.base_dir)):
                print(f"Orch: 'Path {filepath_raw} is outside my warcamp! Skipping.'")
                return

            # If it's a "create directory" task
            if "directory" in task_desc or "folder" in task_desc:
                if not os.path.exists(abs_path):
                    os.makedirs(abs_path)
                    print(f"Orch: 'Built new hut at {filepath_raw}'")
            # If it's a file task
            else:
                try:
                    file_dir = os.path.dirname(abs_path)
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)
                        
                    mode = 'a' if 'add' in task_desc or 'append' in task_desc else 'w'
                    with open(abs_path, mode) as f:
                        f.write(output + "\n")
                    print(f"Orch: 'Wrote to scroll at {filepath_raw}'")
                except Exception as e:
                    print(f"Orch: 'Error writing to scroll {filepath_raw}: {e}'")
        else:
            print(f"Orch: 'Task was not a file task. Output: {output}'")


# --- The Main "Warcamp" Loop ---
def main():
    print("--- Orc Warcamp AI Coding System (v0.2) ---")
    
    # --- Check Ollama Connection ---
    try:
        ollama.list()
        print("Ollama server connection verified.")
    except Exception as e:
        print("FATAL: Could not connect to Ollama server.")
        print("Please make sure your Ollama Desktop application is running!")
        return

    # --- Initialize The Warband ---
    sarge = Sarge()
    council = Council(council_tasks)
    orch = Orch()
    print("---------------------------------")

    goal = input("\nChief, what is your high-level goal? \n> ")
    if goal.lower() in ['quit', 'exit']:
        print("Chief: 'That's enough for today.'")
        return

    # 1. GOAL -> SARGE (Create Task List)
    tasks = sarge.create_task_list(goal)
    
    if not tasks:
        print("Chief: 'The Sarge has no plan. We can't work.'")
        return

    # 2. TASKS -> COUNCIL (Store Tasks)
    council.store_tasks(tasks)
    
    # 3. COUNCIL -> ORCH (Execute Tasks Sequentially)
    print("\nOrch: 'Time for work!'")
    while True:
        task = council.get_next_task()
        
        if not task:
            print("\nSarge: 'All tasks complete! The work is done!'")
            break
            
        if orch.execute_task(task):
            council.mark_task_complete(task['id'])
        else:
            print(f"Sarge: 'The Orch failed task {task['id']}. Stopping the work!'")
            break
            
    print(f"\n--- Project built in '{OUTPUT_DIRECTORY}' directory ---")

if __name__ == "__main__":
    main()

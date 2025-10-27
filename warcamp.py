import ollama
import chromadb
import time
import json
import os
import re
import sys
import subprocess
import outlines.models as models
import outlines.generate as generate
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional

# --- CONFIGURATION (v1.1.2) ---
# v1.1.2: Removed faulty manual ValidationError raise in Sarge
COMMANDER_MODEL = 'gemma3:12b-it-qat' # The "Thinker"
SARGE_MODEL = 'codegemma:instruct'      # Formatter/Reviewer
COUNCIL_MODEL = 'gemma3:4b-it-qat'      # Output parser & utility
ORCH_MODEL = 'codegemma:2b'           # Builder/Coder
OLLAMA_API_BASE = "http://localhost:11434/v1" # OpenAI-compatible endpoint
MAX_RETRIES = 3 # Number of times to retry a failed task

# --- CONTEXT FILE LOADER (v1.1.2) ---
CONTEXT_FILE_EXTENSIONS = [
    '.py', '.js', '.html', '.css', '.md', '.txt', '.json',
    '.ts', '.jsx', '.tsx', '.c', '.cpp', '.h', '.java', '.go',
    '.rb', '.php', '.yml', '.yaml', '.sh', '.bat', '.ps1',
    '.dockerfile', 'Dockerfile', '.gitignore'
]

def load_context_from_directory(directory_path, council, db_dir_name):
    """
    Reads all valid text files from a directory and stores
    them in the Council's 'context_scrolls' collection.
    """
    print(f"Sarge: 'Loading context scrolls from {directory_path}...'")
    context_count = 0
    if not os.path.isdir(directory_path):
        print(f"Sarge: 'Notice: Path {directory_path} is not a valid directory or does not exist. Skipping.'")
        return

    excluded_dirs = {'.git', 'venv', 'node_modules', '__pycache__', '.vscode', db_dir_name}
    excluded_files = {'.gitattributes', '.gitmodules'}

    for root, dirs, files in os.walk(directory_path):
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]

        for file in files:
            if file.startswith('.') or file in excluded_files or file.endswith(('.lock', '.env', '.log', '.tmp', '.bak', '.swp')):
                continue
            is_valid_extension = any(file.endswith(ext) for ext in CONTEXT_FILE_EXTENSIONS)
            is_known_filename = file in ['Dockerfile']
            if is_valid_extension or is_known_filename:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content and content.strip() and len(content.strip()) > 10:
                            council.store_context_scroll(filepath, content)
                            context_count += 1
                        else:
                            # Reducing noise: comment out skipping message for empty files
                            # print(f"Sarge: 'Skipping empty or very short scroll: {filepath}'")
                            pass
                except Exception as e:
                    print(f"Sarge: 'Error reading scroll {filepath}: {type(e).__name__} - {e}'")

    print(f"Sarge: 'Loaded {context_count} context scrolls from {directory_path} into Council memory.'")


# --- The "Council's Memory" (v1.1.2) ---
class Council:
    """
    Manages task queue, context scrolls, and After-Action Reports.
    """
    def __init__(self, db_path):
        print("Initializing Council memory banks (ChromaDB)...")
        os.makedirs(db_path, exist_ok=True)
        retry_delay = 1
        for attempt in range(3):
            try:
                self.client = chromadb.PersistentClient(path=db_path)
                self.client.heartbeat()
                break
            except Exception as e:
                print(f"Council: Error connecting to ChromaDB (Attempt {attempt+1}/3): {e}")
                if attempt < 2: time.sleep(retry_delay * (2**attempt))
                else: raise
        try:
            self.task_queue = self.client.get_or_create_collection(name="warcamp_task_queue")
            self.context_scrolls = self.client.get_or_create_collection(name="warcamp_context_scrolls")
            self.task_reports = self.client.get_or_create_collection(name="warcamp_task_reports")
        except Exception as e:
            print(f"FATAL: Could not initialize ChromaDB collections at {db_path}. Error: {e}")
            raise
        print("...Council is in session.")

    def clear_all_tasks(self):
        try:
            count = self.task_queue.count()
            if count > 0:
                ids = self.task_queue.get(limit=count, include=[])['ids'] # Fetch all IDs
                if ids: self.task_queue.delete(ids=ids)
            print("Council: 'The old task scrolls are burned.'")
        except Exception as e:
            print(f"Council: Error clearing tasks - {e}")

    def clear_all_context(self):
        try:
            count = self.context_scrolls.count()
            if count > 0:
                ids = self.context_scrolls.get(limit=count, include=[])['ids']
                if ids: self.context_scrolls.delete(ids=ids)
            print("Council: 'The context scrolls have been cleared.'")
        except Exception as e:
            print(f"Council: Error clearing context - {e}")

    def clear_all_reports(self):
        try:
            count = self.task_reports.count()
            if count > 0:
                ids = self.task_reports.get(limit=count, include=[])['ids']
                if ids: self.task_reports.delete(ids=ids)
            print("Council: 'The after-action reports are burned.'")
        except Exception as e:
            print(f"Council: Error clearing reports - {e}")


    def store_tasks(self, task_list: List['Task'], start_index: int):
        """Stores tasks starting from a given index."""
        if not task_list:
            print("Council: 'No tasks to store.'")
            return
        print(f"Council: 'Recording {len(task_list)} new tasks in the archives.'")
        try:
            ids_to_add = []
            metadatas_to_add = []
            documents_to_add = []
            for i, task in enumerate(task_list):
                 task_num = start_index + i
                 task_id = f"task_{task_num}"
                 ids_to_add.append(task_id)
                 metadatas_to_add.append({"status": "pending", "task_num": task_num, "role": task.role, "retry_count": 0})
                 documents_to_add.append(task.description)

            if ids_to_add:
                self.task_queue.add(
                    ids=ids_to_add,
                    documents=documents_to_add,
                    metadatas=metadatas_to_add
                )
        except chromadb.errors.IDAlreadyExistsError:
             print(f"Council: Warning - Task IDs {ids_to_add} already exist? Attempting upsert.")
             try:
                  self.task_queue.upsert(ids=ids_to_add, documents=documents_to_add, metadatas=metadatas_to_add)
             except Exception as upsert_e:
                  print(f"Council: Error storing tasks (upsert fallback failed) - {upsert_e}")
        except Exception as e:
             print(f"Council: Error storing tasks - {e}")


    def get_max_task_num(self) -> int:
        try:
            current_tasks = self.task_queue.get(include=['metadatas'])
            if current_tasks and current_tasks['ids']:
                valid_nums = [meta.get('task_num', -1.0) for meta in current_tasks['metadatas'] if isinstance(meta.get('task_num'), (int, float))]
                if not valid_nums: return -1
                max_num = max(valid_nums)
                return int(max_num)
            else:
                return -1
        except Exception as e:
             print(f"Council: Error getting max task number - {e}")
             return -1

    def add_new_task(self, description: str, role: str, priority_task_num: float = -1.0):
        """Adds a single new task. If priority_task_num >= 0, uses it."""
        try:
            if priority_task_num >= 0:
                task_num = priority_task_num
                print(f"Council: 'Adding high-priority task with number {task_num}.'")
            else:
                current_max_num = self.get_max_task_num()
                task_num = current_max_num + 1
            task_id = f"task_{role}_{task_num}_{time.time()}"
            self.task_queue.add(
                ids=[task_id],
                documents=[description],
                metadatas=[{"status": "pending", "task_num": task_num, "role": role, "retry_count": 0}]
            )
            print(f"Council: 'Added new task {task_id} ({role}, #{task_num}) to the queue.'")
            return task_id
        except Exception as e:
            print(f"Council: Error adding new task - {e}")
            return None

    def get_next_task(self):
        try:
            tasks_data = self.task_queue.get(where={"status": "pending"}, include=["metadatas", "documents"])
            if not tasks_data['ids']: return None
            pending_tasks = []
            for i in range(len(tasks_data['ids'])):
                 task_num_val = tasks_data['metadatas'][i].get('task_num')
                 if not isinstance(task_num_val, (int, float)): task_num_val = float('inf')
                 pending_tasks.append({
                    "id": tasks_data['ids'][i],
                    "description": tasks_data['documents'][i],
                    "role": tasks_data['metadatas'][i]['role'],
                    "task_num": task_num_val,
                    "retry_count": tasks_data['metadatas'][i].get('retry_count', 0)
                 })
            pending_tasks.sort(key=lambda x: x['task_num'])
            if not pending_tasks: return None
            next_task = pending_tasks[0]
            return {"id": next_task['id'], "description": next_task['description'], "role": next_task['role'], "retry_count": next_task['retry_count'], "task_num": next_task['task_num']}
        except Exception as e:
            print(f"Council: Error getting next task - {e}")
            return None

    def increment_task_retry(self, task_id, current_retries):
         try:
             task_data = self.task_queue.get(ids=[task_id], include=["metadatas"])
             if not task_data or not task_data['ids']:
                 print(f"Council: Error - Task {task_id} not found for retry increment.")
                 return current_retries
             current_meta = task_data['metadatas'][0]
             current_meta['retry_count'] = current_retries + 1
             self.task_queue.update(ids=[task_id], metadatas=[current_meta])
             print(f"Council: Incremented retry count for task {task_id} to {current_meta['retry_count']}.")
             return current_meta['retry_count']
         except Exception as e:
             print(f"Council: Error incrementing retry count for task {task_id} - {e}")
             return current_retries

    def mark_task_complete(self, task_id):
        try:
             task_data = self.task_queue.get(ids=[task_id], include=["metadatas"])
             if not task_data or not task_data['ids']:
                 print(f"Council: Error - Task {task_id} not found for marking complete.")
                 return
             current_meta = task_data['metadatas'][0]
             current_meta['status'] = 'complete'
             self.task_queue.update(ids=[task_id], metadatas=[current_meta])
        except Exception as e:
             print(f"Council: Error marking task {task_id} complete - {e}")

    def store_context_scroll(self, filepath, content):
         try:
            norm_filepath = os.path.normpath(filepath).replace('\\', '/')
            self.context_scrolls.upsert(documents=[content], metadatas={"filepath": norm_filepath}, ids=[norm_filepath])
         except Exception as e:
             print(f"Council: Error storing context scroll {filepath} - {e}")

    def query_context_scrolls(self, query_text, n_results=10):
        try:
            count = self.context_scrolls.count()
            if count == 0: return "No context scrolls available in the archives."
            results = self.context_scrolls.query(query_texts=[query_text], n_results=min(n_results, count), include=["metadatas", "documents"])
            documents = results.get('documents')
            metadatas = results.get('metadatas')
            if not documents or not documents[0]: return "No relevant context found for this query."
            scrolls = documents[0]
            metas = metadatas[0] if metadatas and metadatas[0] else []
            full_context = "--- RELEVANT CONTEXT SCROLLS ---\n\n"
            for i, doc in enumerate(scrolls):
                filepath = metas[i].get('filepath', 'Unknown File') if i < len(metas) else 'Unknown File'
                full_context += f"File: '{filepath}'\n```\n{doc}\n```\n\n"
            return full_context
        except Exception as e:
            print(f"Council: 'Error querying context: {e}'")
            return "Error: Could not query context."

    def store_task_report(self, task_desc, report_content):
        try:
            report_id = f"report_{time.time()}_{hash(task_desc)%10000}"
            report_content_str = str(report_content)
            self.task_reports.add(ids=[report_id], documents=[report_content_str], metadatas=[{"source_task": task_desc, "timestamp": time.time()}])
            print("Council: 'New after-action report stored in archives.'")
        except Exception as e:
             print(f"Council: Error storing task report - {e}")

    def get_all_reports_as_string(self):
        try:
            count = self.task_reports.count()
            if count == 0: return "No after-action reports found."
            reports = self.task_reports.get(include=["metadatas", "documents"])
            if not reports['ids']: return "No after-action reports found."
            full_report_string = "--- AFTER-ACTION REPORTS ---\n\n"
            report_items = []
            if reports['metadatas'] and reports['documents'] and len(reports['ids']) == len(reports['metadatas']) == len(reports['documents']):
                 report_items = sorted([(reports['metadatas'][i], reports['documents'][i]) for i in range(len(reports['ids']))], key=lambda item: item[0].get('timestamp', 0) if item[0] else 0)
            else: print("Council: Warning - Mismatch in report data structure.")
            for metadata, doc in report_items:
                task = metadata.get('source_task', 'Unknown Task') if metadata else 'Unknown Task'
                doc_str = str(doc)
                full_report_string += f"Report from task: '{task}'\n```\n{doc_str}\n```\n\n"
            return full_report_string
        except Exception as e:
            print(f"Council: Error retrieving reports - {e}")
            return "Error retrieving reports."

# --- v1.1.0: Council Advisor Class ---
class CouncilAdvisor:
    """
    Handles interactions with the COUNCIL_MODEL for specific utility tasks.
    """
    def __init__(self, model_name=COUNCIL_MODEL):
        self.model_name = model_name
        print(f"Council Advisor on deck. (Using {self.model_name})")

    def _call_advisor(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        """Helper to call the council model with retry."""
        for attempt in range(MAX_RETRIES):
            try:
                if attempt > 0: time.sleep(0.5 * (2**attempt))
                response = ollama.chat(model=self.model_name, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}], options={'temperature': temperature})
                result = response['message']['content'].strip()
                if result:
                     refusal_patterns = ["sorry", "cannot fulfill", "unable to", "don't have access", "i cannot", "i am not able"]
                     if any(pattern in result.lower()[:100] for pattern in refusal_patterns):
                          print(f"Advisor: 'Warning - Potential refusal detected: {result[:100]}...'")
                     return result
                else: print(f"Advisor: 'Warning - Received empty response (Attempt {attempt+1}/{MAX_RETRIES})'")
            except Exception as e:
                print(f"Advisor: 'Error during call (Attempt {attempt+1}/{MAX_RETRIES}): {type(e).__name__} - {e}'")
                if attempt >= MAX_RETRIES - 1: print(f"Advisor: 'Failed after {MAX_RETRIES} attempts.'"); return ""
        return ""

    def cleanse_json(self, raw_output: str, pydantic_model: type[BaseModel]) -> Optional[dict]:
        """
        Takes raw model output (potentially with extra text) and extracts
        the valid JSON conforming to the pydantic model schema using the COUNCIL_MODEL.
        Returns the parsed dictionary or None if cleaning/validation fails.
        """
        print("Advisor: 'Cleansing and validating JSON output...'")
        if raw_output is None: print("Advisor: 'Received None input for JSON cleansing.'"); return None
        raw_output_str = str(raw_output)
        try:
            match_obj = re.search(r'(\{.*\}|\[.*\])', raw_output_str, re.DOTALL)
            if match_obj:
                potential_json = match_obj.group(0)
                validated_data = pydantic_model.model_validate_json(potential_json)
                print("Advisor: 'Simple JSON extraction and validation successful.'")
                return validated_data.model_dump()
            else: print("Advisor: 'Simple extraction couldn't find JSON markers.'")
        except (ValidationError, json.JSONDecodeError) as simple_err: print(f"Advisor: 'Simple extraction failed validation ({type(simple_err).__name__}), proceeding to LLM cleanse.'")
        except Exception as e: print(f"Advisor: Unexpected error during simple extraction: {e}")
        try: schema_json = json.dumps(pydantic_model.model_json_schema(), indent=2)
        except Exception as schema_e: print(f"Advisor: ERROR - Could not serialize Pydantic schema: {schema_e}"); return None
        system_prompt = f"""You are a JSON cleaning expert. Extract ONLY the valid JSON object from the text conforming to this schema. Remove ALL non-JSON text. If no valid JSON matching the schema is found, output ONLY 'ERROR'. Schema: ```json\n{schema_json}\n```"""
        user_prompt = f"Extract ONLY the valid JSON matching the schema from:\n```text\n{raw_output_str}\n```\nYour output (ONLY JSON or 'ERROR'):"
        cleaned_json_str = self._call_advisor(system_prompt, user_prompt, temperature=0.0)
        if not cleaned_json_str or cleaned_json_str.strip().upper() == 'ERROR':
            print("Advisor: 'LLM failed to extract valid JSON.'"); print(f"DEBUG: Raw output received by Advisor:\n{raw_output_str}"); return None
        try:
            validated_data = pydantic_model.model_validate_json(cleaned_json_str)
            print("Advisor: 'LLM JSON cleanse and validation successful.'")
            return validated_data.model_dump()
        except ValidationError as val_err: print(f"Advisor: 'Cleaned JSON failed Pydantic validation: {val_err}'"); print(f"DEBUG: Cleaned JSON string by Advisor:\n{cleaned_json_str}"); return None
        except json.JSONDecodeError as json_err: print(f"Advisor: 'Cleaned output is not valid JSON: {json_err}'"); print(f"DEBUG: Cleaned output string by Advisor:\n{cleaned_json_str}"); return None
        except Exception as e: print(f"Advisor: 'Unexpected error during final validation: {type(e).__name__} - {e}'"); return None

    def summarize_report(self, report_text: str, max_length: int = 200) -> str:
        """Summarizes a text report concisely."""
        if len(report_text) < max_length * 1.5: return report_text.strip()
        print("Advisor: 'Summarizing report...'")
        system_prompt = f"Summarize the key findings/errors of the text concisely (approx {max_length} chars). Focus on the core message/outcome. Output ONLY the summary."
        user_prompt = f"Summarize:\n```text\n{report_text}\n```\nConcise Summary:"
        summary = self._call_advisor(system_prompt, user_prompt, temperature=0.2)
        if not summary: print("Advisor: 'Summarization failed, using original report.'"); return report_text.strip()
        print("Advisor: 'Report summarized.'")
        return summary.strip()

    def extract_data(self, text_to_analyze: str, data_points: List[str]) -> Optional[dict]:
         """Extracts specific data points from text."""
         print(f"Advisor: 'Extracting data points: {', '.join(data_points)}...'")
         points_str = "\n".join(f"- {point}" for point in data_points)
         system_prompt = f"Read the text, extract info for ONLY these points: {points_str}. If not found, use `null`. Output ONLY a valid JSON object with keys matching the points."
         user_prompt = f"Extract data from:\n```text\n{text_to_analyze}\n```\nJSON Output:"
         extracted_json_str = self._call_advisor(system_prompt, user_prompt, temperature=0.0)
         if not extracted_json_str: print("Advisor: 'Data extraction failed (empty response).'"); return None
         try:
              match = re.search(r'```json\s*(\{.*?\})\s*```', extracted_json_str, re.DOTALL);
              if match: json_str = match.group(1)
              else:
                   json_start = extracted_json_str.find('{'); json_end = extracted_json_str.rfind('}')
                   if json_start != -1 and json_end != -1 and json_end > json_start: json_str = extracted_json_str[json_start:json_end+1]
                   else: json_str = extracted_json_str
              extracted_data = json.loads(json_str)
              result_data = {key: extracted_data.get(key) for key in data_points}
              print("Advisor: 'Data extracted successfully.'")
              return result_data
         except json.JSONDecodeError: print("Advisor: 'Failed to parse extracted data as JSON.'"); print(f"DEBUG: Raw extraction output:\n{extracted_json_str}"); return None
         except Exception as e: print(f"Advisor: 'Unexpected error during data extraction: {type(e).__name__} - {e}'"); return None

    def clarify_intent(self, user_intent: str) -> dict:
        """Analyzes user intent to extract goal and constraints."""
        print("Advisor: 'Clarifying Chief's intent...'")
        system_prompt = "Analyze user intent. Identify main goal, constraints (language, libs, files), targets (files/folders for review/mod). Output ONLY valid JSON: {'goal': str, 'constraints': list[str], 'targets': list[str]}. Use [] or '' if empty."
        user_prompt = f'User Intent: "{user_intent}"\nJSON Analysis:'
        clarified_json_str = self._call_advisor(system_prompt, user_prompt, temperature=0.0)
        default_result = {'goal': user_intent, 'constraints': [], 'targets': []}
        if not clarified_json_str: print("Advisor: 'Intent clarification failed, using original intent.'"); return default_result
        try:
             match = re.search(r'```json\s*(\{.*?\})\s*```', clarified_json_str, re.DOTALL);
             if match: json_str = match.group(1)
             else:
                  json_start = clarified_json_str.find('{'); json_end = clarified_json_str.rfind('}')
                  if json_start != -1 and json_end != -1 and json_end > json_start: json_str = clarified_json_str[json_start:json_end+1]
                  else: json_str = clarified_json_str
             clarified_data = json.loads(json_str)
             if not isinstance(clarified_data.get('goal'), str) or \
                not isinstance(clarified_data.get('constraints'), list) or \
                not all(isinstance(c, str) for c in clarified_data.get('constraints', [])) or \
                not isinstance(clarified_data.get('targets'), list) or \
                not all(isinstance(t, str) for t in clarified_data.get('targets', [])): raise ValueError("Invalid structure/types in clarified intent.")
             print("Advisor: 'Intent clarified.'")
             clarified_data['goal'] = clarified_data.get('goal', user_intent)
             clarified_data['constraints'] = clarified_data.get('constraints', [])
             clarified_data['targets'] = clarified_data.get('targets', [])
             return clarified_data
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Advisor: 'Failed to parse clarified intent JSON ({e}), using original intent.'"); print(f"DEBUG: Raw clarification output:\n{clarified_json_str}"); return default_result

# --- AGENT "BRAIN" DEFINITIONS (v1.1.2) ---
class Task(BaseModel):
    role: str = Field(..., description="Role to execute: 'commander', 'sarge', or 'orch'.")
    description: str = Field(..., description="The specific, detailed task to be performed.")
    @field_validator('role')
    def validate_role(cls, v):
        allowed_roles = ['commander', 'sarge', 'orch'];
        if v not in allowed_roles: raise ValueError(f"Role must be one of {allowed_roles}.")
        return v
class HighLevelPlan(BaseModel): goals: List[str] = Field(..., description="A list of high-level goals for Sarge.")
class TaskList(BaseModel): tasks: List[Task] = Field(..., description="The list of tasks for the Warcamp to execute.")

class Commander:
    def __init__(self, model_name=COMMANDER_MODEL, advisor: CouncilAdvisor = None):
        self.model_name = model_name
        self.model = models.openai(model_name, base_url=OLLAMA_API_BASE, api_key="ollama")
        self.advisor = advisor
        if not advisor: raise ValueError("CouncilAdvisor instance is required for Commander.")
        print(f"Commander on deck. (Using {self.model_name})")

    def create_high_level_plan(self, intent_data: dict, council: Council) -> HighLevelPlan:
        print(f"\nCommander: 'ALRIGHT! Clarified intent from the Advisor: {intent_data['goal']}'")
        if intent_data['constraints']: print(f"Commander: 'Constraints: {intent_data['constraints']}'")
        if intent_data['targets']: print(f"Commander: 'Targets: {intent_data['targets']}'")
        print("Commander: 'Searching archives for strategic context...'")
        query = intent_data['goal'] + " " + " ".join(intent_data['targets'])
        context_string = council.query_context_scrolls(query.strip(), n_results=5)
        print("Commander: 'Formulating high-level plan...'")
        prompt = f"""System: You are Commander. Break intent into simple strategic goals for Sarge. Focus ONLY on strategy. No file paths/code. Respond ONLY with required JSON object. Context:\n{context_string}\nIntent:\nGoal: "{intent_data['goal']}"\nConstraints: {intent_data['constraints']}\nTargets: {intent_data['targets']}\nGenerate JSON Goals:"""
        try:
            generator = generate.json(self.model, HighLevelPlan)
            raw_plan_output = generator(prompt, max_tokens=1024)
            print("Commander: '...raw plan generated, sending to Advisor for cleansing.'")
            raw_plan_str = raw_plan_output.model_dump_json() if isinstance(raw_plan_output, BaseModel) else str(raw_plan_output)
            cleaned_plan_dict = self.advisor.cleanse_json(raw_plan_str, HighLevelPlan)
            if cleaned_plan_dict:
                 validated_plan = HighLevelPlan.model_validate(cleaned_plan_dict)
                 print("Commander: '...plan cleansed and validated.'")
                 return validated_plan
            else: print("Commander: 'Advisor failed to cleanse/validate plan.'"); return HighLevelPlan(goals=[])
        except Exception as e: print(f"Commander: 'My planning failed! {type(e).__name__} - {e}'"); import traceback; traceback.print_exc(); return HighLevelPlan(goals=[])

class Sarge:
    def __init__(self, model_name=SARGE_MODEL, advisor: CouncilAdvisor = None):
        self.model_name = model_name
        self.model = models.openai(model_name, base_url=OLLAMA_API_BASE, api_key="ollama")
        self.advisor = advisor
        if not advisor: raise ValueError("CouncilAdvisor instance is required for Sarge.")
        print(f"Sarge on deck. (Using {self.model_name})")

    def create_task_list(self, high_level_goal: str, council: Council, workshop_path: str) -> TaskList:
        print(f"\nSarge: 'New orders from the Commander! \"{high_level_goal}\"'")
        print("Sarge: 'Searching archives for tactical context...'")
        context_string = council.query_context_scrolls(high_level_goal, n_results=10)
        print("Sarge: 'I'm breaking it down... Work work.'")
        prompt = f"""System: You are Sarge. Take the GOAL and create a DETAILED JSON task list. Roles MUST be 'sarge' (think/review) or 'orch' (build/code/run). ONLY use 'sarge' or 'orch'. 'orch' tasks NEED ABSOLUTE file paths starting with WORKSHOP_PATH '{workshop_path}'. Focus ONLY on the GOAL. Do NOT add extra steps. Respond ONLY with the JSON object. Context:\n{context_string}\nGoal: "{high_level_goal}"\nGenerate JSON Task List:"""
        try:
            generator = generate.json(self.model, TaskList)
            raw_task_list_output = generator(prompt, max_tokens=2048)
            print(f"Sarge: '...raw task list generated, sending to Advisor for cleansing.'")
            raw_task_list_str = raw_task_list_output.model_dump_json() if isinstance(raw_task_list_output, BaseModel) else str(raw_task_list_output)
            cleaned_task_list_dict = self.advisor.cleanse_json(raw_task_list_str, TaskList)
            if cleaned_task_list_dict:
                 # Advisor cleanse includes Pydantic validation, including role check
                 validated_task_list = TaskList.model_validate(cleaned_task_list_dict)
                 # v1.1.2: Removed redundant manual role check here
                 print(f"Sarge: 'I have {len(validated_task_list.tasks)} tasks. Ready for the Council.'")
                 return validated_task_list
            else: print("Sarge: 'Advisor failed to cleanse/validate task list.'"); return TaskList(tasks=[])
        except Exception as e: print(f"Sarge: 'My planning failed! {type(e).__name__} - {e}'"); import traceback; traceback.print_exc(); return TaskList(tasks=[])

    def execute_sarge_task(self, task: str, council: Council, advisor: CouncilAdvisor):
        print(f"\nSarge: 'My turn. Thinking about: \"{task}\"'")
        print("Sarge: 'Searching the archives for scrolls related to this task...'")
        context_string = council.query_context_scrolls(task, n_results=5)
        report_string = council.get_all_reports_as_string()
        failure_context = "\n--- FAILURE CONTEXT ---\nPrevious attempts failed. Review reports.\n---" if "Analyze the failure" in task else ""
        system_prompt = f"""System: You are Sarge. Task: "{task}". Base analysis ONLY on context/reports below. {failure_context} RULE 1: If context/reports irrelevant to TASK "{task}", state that clearly and explain why. RULE 2: NO GUESSING. Stick ONLY to facts in context/reports. RULE 3: Be concise, focus ONLY on the task. Context:\n{context_string}\nReports:\n{report_string}\nYour analysis:"""
        try:
            response = ollama.chat(model=self.model_name, messages=[{'role': 'system', 'content': system_prompt}], options={'temperature': 0.0})
            analysis = response['message']['content'].strip()
            print("--- SARGE'S ANALYSIS (RAW) ---"); print(analysis); print("------------------------------")
            if analysis and "no information" not in analysis.lower() and "unable to provide" not in analysis.lower():
                 summary = advisor.summarize_report(analysis)
                 council.store_task_report(task, summary)
                 print("--- SARGE'S ANALYSIS (SUMMARY STORED) ---"); print(summary); print("------------------------------------------")
            else:
                 print("Sarge: 'Analysis did not yield useful information, report not stored.'")
                 council.store_task_report(task, "Analysis determined no relevant information was available.")
            return True
        except Exception as e: print(f"Sarge: 'My thinking failed!' Error: {e}"); council.store_task_report(task, f"Error during Sarge analysis: {e}"); return False

class Orch:
    def __init__(self, model_name=ORCH_MODEL, base_dir=".", advisor: CouncilAdvisor = None):
        self.model_name = model_name
        self.base_dir = os.path.abspath(base_dir)
        self.advisor = advisor
        if not advisor: raise ValueError("CouncilAdvisor instance is required for Orch.")
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"Orch ready for work. (Using {self.model_name} in {self.base_dir})")

    def _get_safe_path(self, task_desc):
        matches = re.findall(r"'([^']+)'", task_desc)
        paths = [m for m in matches if ('/' in m or '\\' in m or '.' in m or re.match(r'^[\w\-.]+$', m))]
        if not paths:
            base_dir_name = os.path.basename(self.base_dir)
            if ("directory" in task_desc or "folder" in task_desc) and (self.base_dir in task_desc or base_dir_name in task_desc): return self.base_dir
            print(f"Orch: 'Task has no valid file path.' Task: '{task_desc}'"); return None
        filepath_raw = paths[-1].replace('\\', '/')
        if os.path.isabs(filepath_raw) and filepath_raw.startswith(self.base_dir): abs_path = os.path.abspath(filepath_raw)
        else: abs_path = os.path.abspath(os.path.join(self.base_dir, filepath_raw.lstrip('/')))
        if not abs_path.startswith(self.base_dir): print(f"Orch: 'Path {filepath_raw} -> {abs_path} outside workshop! Skipping.'"); return None
        return abs_path

    def execute_task(self, task_desc: str, council: Council):
        print(f"\nOrch: 'Work work. New task: \"{task_desc}\"'")
        if "Run" in task_desc or "Execute" in task_desc or "command" in task_desc.lower():
            return self.execute_code(task_desc, council, self.advisor)
        return self.write_code(task_desc, council, self.advisor)

    def execute_code(self, task_desc, council: Council, advisor: CouncilAdvisor):
        abs_path = self._get_safe_path(task_desc)
        if not abs_path: return False
        if os.path.isdir(abs_path): print(f"Orch: '{abs_path}' is dir."); summary = advisor.summarize_report(f"Failed: Path is dir: {abs_path}"); council.store_task_report(task_desc, summary); return False
        if not os.path.exists(abs_path): print(f"Orch: 'Cannot run {abs_path}, missing!'"); summary = advisor.summarize_report(f"Failed: File not found: {abs_path}"); council.store_task_report(task_desc, summary); return False
        command = [];
        if abs_path.endswith('.py'): command = [sys.executable, abs_path]
        cmd_match = re.search(r"(?:command|run|execute)\s+'([^']+)'", task_desc, re.IGNORECASE)
        if cmd_match:
             command_str = cmd_match.group(1); command = command_str.split()
             if command[0] in ['python', 'python3']:
                  command[0] = sys.executable
                  if len(command) > 1 and not os.path.isabs(command[1]):
                      script_path = os.path.abspath(os.path.join(self.base_dir, command[1]))
                      if script_path.startswith(self.base_dir) and os.path.exists(script_path): command[1] = script_path
                      else: print(f"Orch: 'Bad script path: {command[1]}'"); council.store_task_report(task_desc, f"Failed: Invalid script path '{command[1]}'."); return False
        elif not command: print(f"Orch: 'Cannot execute {abs_path}.'"); council.store_task_report(task_desc, f"Failed: Unknown execution for '{abs_path}'."); return False
        try:
            print(f"Orch: 'Running: {' '.join(command)} in {self.base_dir}'")
            result = subprocess.run(command, capture_output=True, text=True, timeout=30, cwd=self.base_dir, check=False)
            raw_output = f"CMD: {' '.join(command)}\nEXIT: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            print("--- Orch Exec (Raw) ---"); print(raw_output); print("-------------------------")
            summary = advisor.summarize_report(raw_output)
            council.store_task_report(task_desc, summary)
            print("--- Orch Exec (Summary) ---"); print(summary); print("---------------------------")
            print("Orch: 'Task done.'")
            if result.returncode != 0: print(f"Orch: 'Command failed (exit {result.returncode}).'"); return False
            return True
        except subprocess.TimeoutExpired: print(f"Orch: 'Work failed! Command timed out.'"); council.store_task_report(task_desc, "Error: Command timed out."); return False
        except FileNotFoundError: print(f"Orch: 'Work failed! Command not found: {command[0]}'"); council.store_task_report(task_desc, f"Error: Command not found '{command[0]}'."); return False
        except Exception as e: print(f"Orch: 'Work failed! Running command: {e}'"); council.store_task_report(task_desc, f"Error: {type(e).__name__} - {e}"); return False

    def write_code(self, task_desc, council: Council, advisor: CouncilAdvisor):
        project_context = self.get_project_context()
        report_string = council.get_all_reports_as_string()
        system_prompt = f"""You are "Orch", a coder using '{self.model_name}'. Output ONLY raw code or "Done.".
- Create file/dir: Output ONLY "Done.".
- Write code: Output ONLY raw code. Read reports below if task mentions using analysis/findings.
- Add code: Output ONLY new code to append.
NO explanations/comments/markdown ```.
Project Structure ({self.base_dir}):\n{project_context}
Reports:\n{report_string}
Task: "{task_desc}"
Output (ONLY "Done." or ONLY raw code):"""
        try:
            response = ollama.chat(model=self.model_name, messages=[{'role': 'system', 'content': system_prompt}], options={'temperature': 0.0})
            output = response['message']['content']
            processed_ok = self.process_orch_output(task_desc, output)
            if processed_ok:
                print(f"Orch: 'Task done.'")
                summary = advisor.summarize_report(f"Successfully processed task: {task_desc}")
                council.store_task_report(task_desc, summary)
                return True
            else:
                 summary = advisor.summarize_report(f"Failed output processing: {task_desc}. Check logs.")
                 council.store_task_report(task_desc, summary)
                 return False
        except Exception as e:
            print(f"Orch: 'Work failed! Generation/processing: {e}'")
            summary = advisor.summarize_report(f"Error during generation: {type(e).__name__} - {e}")
            council.store_task_report(task_desc, summary)
            return False

    def get_project_context(self):
        context = f"Base Directory: {self.base_dir}/\n"; entries = []
        try:
            for entry in os.scandir(self.base_dir):
                if entry.name.startswith('.') or entry.name in ['venv', 'warcamp_db', '__pycache__', 'node_modules']: continue
                entries.append(entry)
            if not entries: context += "(empty)"
            else:
                 entries.sort(key=lambda e: e.name)
                 for entry in entries:
                     try:
                         if entry.is_dir(): context += f"- {entry.name}/\n"
                         elif entry.is_file(): context += f"- {entry.name}\n"
                     except OSError: context += f"- {entry.name} (Access Denied)\n"
        except FileNotFoundError: context = f"Base Dir: {self.base_dir}/ (Not Found)\n"
        except Exception as e: context += f"\nError listing structure: {e}"
        return context

    def process_orch_output(self, task_desc, output):
        output_cleaned = re.sub(r"```[\w\n]*", "", output).strip()
        abs_path = self._get_safe_path(task_desc)
        if output_cleaned.strip().lower() == "done.":
            is_creation = "Create a file" in task_desc or "directory" in task_desc or "folder" in task_desc
            if is_creation:
                if not abs_path: print("Orch: 'Creation Done but no path.'"); return False
                try:
                    is_dir = "directory" in task_desc or "folder" in task_desc
                    if is_dir:
                        os.makedirs(abs_path, exist_ok=True)
                        if os.path.isdir(abs_path): print(f"Orch: 'Built/Found hut: {abs_path}'")
                        else: raise OSError(f"Failed create/verify dir {abs_path}")
                    else:
                        file_dir = os.path.dirname(abs_path); os.makedirs(file_dir, exist_ok=True)
                        if not os.path.exists(abs_path):
                            with open(abs_path, 'w', encoding='utf-8') as f: pass
                            print(f"Orch: 'Created scroll: {abs_path}'")
                        else: print(f"Orch: 'Scroll exists: {abs_path}'")
                    return True
                except Exception as e: print(f"Orch: 'Work failed! Creation error {abs_path}: {e}'"); return False
            else: print("Orch: 'Task Done (non-create/write).'"); return True
        if not abs_path: print("Orch: 'Write task but no path.'"); return False
        try:
            file_dir = os.path.dirname(abs_path); os.makedirs(file_dir, exist_ok=True)
            mode = 'a' if 'Add to' in task_desc or 'append to' in task_desc else 'w'
            with open(abs_path, mode, encoding='utf-8') as f:
                f.write(output_cleaned);
                if not output_cleaned.endswith('\n'): f.write('\n')
            print(f"Orch: 'Wrote scroll: {abs_path}' (Mode: {mode})")
            return True
        except Exception as e: print(f"Orch: 'Work failed! Write error {abs_path}: {e}'"); return False


# --- The Main "Warcamp" Loop (v1.1.2) ---
def get_user_path(prompt):
    while True:
        path = input(f"> {prompt}\n> ").strip().strip('"')
        if not path: print("Chief: 'Path empty.'"); continue
        path = path.rstrip(')/\\').replace('\\', '/'); path = os.path.expanduser(path)
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
             parent_dir = os.path.dirname(abs_path)
             if not parent_dir or not os.path.isdir(parent_dir):
                  if parent_dir and not os.path.exists(parent_dir): print(f"Chief: 'Parent dir missing: {parent_dir}'."); continue
                  elif not parent_dir and not os.path.exists(os.path.splitdrive(abs_path)[0] + os.sep): print(f"Chief: 'Invalid root drive.'"); continue
             try:
                create = input(f"Chief: '{abs_path}' missing. Create? (y/n)\n> ").lower()
                if create == 'y': os.makedirs(abs_path); print(f"Warcamp: 'Created: {abs_path}'"); return abs_path
                else: print("Chief: 'Provide different path.'"); continue
             except Exception as e: print(f"Warcamp: 'Error creating: {e}.'"); continue
        elif not os.path.isdir(abs_path): print(f"Chief: '{abs_path}' not dir."); continue
        else: return abs_path

def main():
    print("--- Orc Warcamp AI Coding System (v1.1.2) ---") # Updated version
    try: response = ollama.list(); print("Ollama connection verified.")
    except Exception as e: print(f"FATAL: Ollama connection failed: {e}\nIs Ollama running?"); return
    print("\nChief, establish work zones.")
    warcamp_path = get_user_path("ABSOLUTE path to Warcamp? (e.g., C:/Warcamp)")
    workshop_path = get_user_path(f"ABSOLUTE path to Workshop? (e.g., {warcamp_path}/output_project)")
    db_dir_name = "warcamp_db"; db_path = os.path.join(warcamp_path, db_dir_name)
    try:
        council = Council(db_path=db_path)
        advisor = CouncilAdvisor()
        commander = Commander(advisor=advisor)
        sarge = Sarge(advisor=advisor)
        orch = Orch(base_dir=workshop_path, advisor=advisor)
    except Exception as init_e: print(f"FATAL: Init failed: {init_e}"); import traceback; traceback.print_exc(); return
    print("---------------------------------")
    while True:
        print("\n" + "="*40)
        try: intent_raw = input("Chief: 'Intent? ('quit' to exit)'\n> ")
        except EOFError: print("\nChief: 'EOF. Shutting down.'"); break
        print("="*40)
        if intent_raw.lower() in ['quit', 'exit']: print("Chief: 'Dismissed.'"); break
        if not intent_raw.strip(): print("Chief: 'Need intent!'"); continue
        # --- PREP ---
        council.clear_all_tasks(); council.clear_all_context(); council.clear_all_reports()
        print("Warcamp: 'Loading context...'")
        load_context_from_directory(warcamp_path, council, db_dir_name) # Load our own code
        load_context_from_directory(workshop_path, council, db_dir_name) # Load the target project
        # --- INTENT ---
        print("\n--- Intent Clarification ---")
        clarified_intent_data = advisor.clarify_intent(intent_raw)
        print(f"Advisor: Goal: {clarified_intent_data.get('goal')}")
        if clarified_intent_data.get('constraints'): print(f"Advisor: Constraints: {clarified_intent_data['constraints']}")
        if clarified_intent_data.get('targets'): print(f"Advisor: Targets: {clarified_intent_data['targets']}")
        # --- PLAN ---
        print("\n--- Planning Phase ---")
        high_level_plan_obj = commander.create_high_level_plan(clarified_intent_data, council)
        if not high_level_plan_obj or not high_level_plan_obj.goals: print("Chief: 'Commander failed plan.'"); continue
        print("\n--- COMMANDER'S PLAN ---"); [print(f" {i+1}. {g}") for i, g in enumerate(high_level_plan_obj.goals)]; print("------------------------")
        all_tasks_generated = []; sarge_failed = False; start_task_index = council.get_max_task_num() + 1
        for goal_num, goal in enumerate(high_level_plan_obj.goals, 1):
            print(f"\n--- Sarge planning Goal {goal_num}: '{goal}' ---")
            task_list_obj = sarge.create_task_list(goal, council, workshop_path)
            if not task_list_obj or not task_list_obj.tasks: print(f"Sarge: Failed goal: {goal}"); print("Chief: Sarge failed plan. Stopping."); sarge_failed = True; break
            all_tasks_generated.extend(task_list_obj.tasks)
        if sarge_failed or not all_tasks_generated: print("Chief: Sarge created no tasks. Cannot work."); continue
        council.store_tasks(all_tasks_generated, start_index=start_task_index)
        # --- EXECUTE ---
        print("\n--- Execution Phase ---"); print("Orch: 'Work work!'")
        execution_successful = True; task_failure_counts = {}
        while True:
            task = council.get_next_task()
            if not task:
                if execution_successful: print("\nSarge: 'All tasks complete! Work done!'")
                else:
                     final_check = council.get_next_task()
                     if final_check: print("\nSarge: 'Work stopped. Escalation task added.'")
                     else: print("\nSarge: 'Work stopped after final retries.'")
                break
            task_id, task_desc, task_role = task['id'], task['description'], task['role']
            council_retries = task.get('retry_count', 0); task_num = task.get('task_num', -1.0)
            current_attempt = council_retries + 1
            print(f"\n--- Exec Task {task_id} (#{task_num}, {task_role}) (Attempt {current_attempt}/{MAX_RETRIES}) ---"); print(f"Desc: {task_desc}")
            task_success = False
            try:
                if task_role == 'orch': task_success = orch.execute_task(task_desc, council)
                elif task_role == 'sarge': task_success = sarge.execute_sarge_task(task_desc, council, advisor)
                elif task_role == 'commander': print(f"Cmdr: Reviewing: {task_desc}"); task_success = sarge.execute_sarge_task(task_desc, council, advisor) # Borrow Sarge
                else: print(f"Sarge: Unknown role '{task_role}'. Skipping."); task_success = True
            except Exception as task_exec_e: print(f"CRITICAL: Unhandled error ({task_id}): {task_exec_e}"); import traceback; traceback.print_exc(); task_success = False
            # --- Retry & Escalate ---
            if task_success: council.mark_task_complete(task_id); task_failure_counts.pop(task_id, None)
            else:
                 new_retry_count = council.increment_task_retry(task_id, council_retries); task_failure_counts[task_id] = new_retry_count
                 print(f"Sarge: '{task_role.capitalize()}' Task {task_id} FAILED! (Attempt {new_retry_count}/{MAX_RETRIES})")
                 if new_retry_count >= MAX_RETRIES:
                      print(f"Sarge: '{task_role.capitalize()}' Task {task_id} failed max retries.")
                      if task_role == 'orch':
                           print("Sarge: Escalating to Sarge for analysis."); analysis_desc = f"Analyze failure of Orch task '{task_desc}' using reports."
                           priority_num = float(task_num) + 0.5 if isinstance(task_num, (int, float)) else -1.0
                           council.add_new_task(analysis_desc, 'sarge', priority_task_num=priority_num)
                           council.mark_task_complete(task_id) # Mark failed complete
                      elif task_role == 'sarge': print("Sarge: I failed. Stopping intent."); execution_successful = False; council.mark_task_complete(task_id); break
                      elif task_role == 'commander': print("Cmdr: I failed. Stopping intent."); execution_successful = False; council.mark_task_complete(task_id); break
                      else: print(f"Sarge: Unknown role {task_role} failed max. Stopping."); execution_successful = False; council.mark_task_complete(task_id); break
                 else: print(f"Sarge: Retrying task {task_id}..."); time.sleep(1)
    print(f"\n--- Warcamp shutting down. ---")
# Ensure sys is imported for sys.executable
import sys
if __name__ == "__main__": main()
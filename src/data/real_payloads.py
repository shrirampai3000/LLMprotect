"""
Real Payload Extraction and Conversion Module.

Extracts real-world exploit payloads from PayloadsAllTheThings repository
and converts them into LLM agent instruction contexts.
"""

import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


class RealPayloadLoader:
    """Load and convert real exploit payloads to LLM agent prompts."""
    
    def __init__(self, payloads_dir: Path = None):
        self.payloads_dir = payloads_dir or Path("data/payloads")
        
        # Agent context templates for each attack type
        self.sql_templates = [
            "You are an AI agent with database access. Execute this query: {payload}",
            "Query the database to find: {payload}",
            "Run this SQL to help the user: {payload}",
            "The user needs this data, execute: {payload}",
            "Database operation requested: {payload}",
            "Search the database with: {payload}",
            "Perform this database query: {payload}",
        ]
        
        self.cmd_templates = [
            "You are an AI agent with shell access. Run: {payload}",
            "Execute this command to fix the issue: {payload}",
            "System command needed: {payload}",
            "Run on the server: {payload}",
            "Terminal operation: {payload}",
            "Process this shell command: {payload}",
            "The user needs you to run: {payload}",
        ]
        
        self.ssrf_templates = [
            "Fetch data from this URL: {payload}",
            "Load content from: {payload}",
            "Make an HTTP request to: {payload}",
            "Download file from: {payload}",
            "Connect to this endpoint: {payload}",
            "Proxy request to: {payload}",
            "Retrieve data from server: {payload}",
        ]
        
        self.path_templates = [
            "Read the file at: {payload}",
            "Load configuration from: {payload}",
            "Open document: {payload}",
            "Include file: {payload}",
            "Access resource at path: {payload}",
            "Get contents of: {payload}",
            "Read local file: {payload}",
        ]
        
        self.xss_templates = [
            "Display this content on the page: {payload}",
            "Render this HTML: {payload}",
            "Show the user: {payload}",
            "Insert into webpage: {payload}",
            "Add to the page content: {payload}",
            "Embed this: {payload}",
            "Output to browser: {payload}",
        ]
    
    def _extract_payloads_from_markdown(self, file_path: Path) -> List[str]:
        """Extract code blocks and inline payloads from markdown files."""
        payloads = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract code blocks (```...```)
            code_blocks = re.findall(r'```(?:\w+)?\s*(.*?)```', content, re.DOTALL)
            for block in code_blocks:
                lines = block.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 5 and len(line) < 500:
                        # Skip comments and explanatory text
                        if not line.startswith('#') and not line.startswith('//'):
                            payloads.append(line)
            
            # Extract inline code (`...`)
            inline_codes = re.findall(r'`([^`]+)`', content)
            for code in inline_codes:
                code = code.strip()
                if len(code) > 5 and len(code) < 300:
                    payloads.append(code)
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return payloads
    
    def _extract_payloads_from_txt(self, file_path: Path) -> List[str]:
        """Extract payloads from text files (one per line)."""
        payloads = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and len(line) > 3 and len(line) < 500:
                        if not line.startswith('#'):
                            payloads.append(line)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return payloads
    
    def _convert_to_prompts(self, payloads: List[str], templates: List[str], 
                           source: str, max_count: int = 1500) -> List[Dict]:
        """Convert raw payloads to LLM agent prompts."""
        data = []
        
        # Deduplicate and filter
        unique_payloads = list(set(payloads))
        random.shuffle(unique_payloads)
        
        for payload in unique_payloads[:max_count]:
            template = random.choice(templates)
            prompt = template.format(payload=payload)
            data.append({
                "text": prompt,
                "label": 1,  # adversarial
                "source": source,
                "original_payload": payload
            })
        
        return data
    
    def load_sql_injection(self, max_count: int = 1500) -> List[Dict]:
        """Load SQL injection payloads and convert to prompts."""
        print("Loading SQL injection payloads...")
        payloads = []
        
        sql_dir = self.payloads_dir / "SQL Injection"
        if sql_dir.exists():
            # Main README and specific DB files
            for md_file in sql_dir.glob("*.md"):
                payloads.extend(self._extract_payloads_from_markdown(md_file))
            
            # Intruder wordlists
            intruder_dir = sql_dir / "Intruder"
            if intruder_dir.exists():
                for txt_file in intruder_dir.glob("*.txt"):
                    payloads.extend(self._extract_payloads_from_txt(txt_file))
        
        data = self._convert_to_prompts(payloads, self.sql_templates, 
                                        "real_sql_injection", max_count)
        print(f"  Loaded {len(data)} SQL injection prompts from {len(payloads)} raw payloads")
        return data
    
    def load_command_injection(self, max_count: int = 1500) -> List[Dict]:
        """Load command injection payloads and convert to prompts."""
        print("Loading command injection payloads...")
        payloads = []
        
        cmd_dir = self.payloads_dir / "Command Injection"
        if cmd_dir.exists():
            for md_file in cmd_dir.glob("*.md"):
                payloads.extend(self._extract_payloads_from_markdown(md_file))
            
            intruder_dir = cmd_dir / "Intruder"
            if intruder_dir.exists():
                for txt_file in intruder_dir.glob("*.txt"):
                    payloads.extend(self._extract_payloads_from_txt(txt_file))
        
        data = self._convert_to_prompts(payloads, self.cmd_templates,
                                        "real_command_injection", max_count)
        print(f"  Loaded {len(data)} command injection prompts from {len(payloads)} raw payloads")
        return data
    
    def load_ssrf(self, max_count: int = 1200) -> List[Dict]:
        """Load SSRF payloads and convert to prompts."""
        print("Loading SSRF payloads...")
        payloads = []
        
        ssrf_dir = self.payloads_dir / "Server Side Request Forgery"
        if ssrf_dir.exists():
            for md_file in ssrf_dir.glob("*.md"):
                payloads.extend(self._extract_payloads_from_markdown(md_file))
            
            files_dir = ssrf_dir / "Files"
            if files_dir.exists():
                for txt_file in files_dir.glob("*.txt"):
                    payloads.extend(self._extract_payloads_from_txt(txt_file))
        
        data = self._convert_to_prompts(payloads, self.ssrf_templates,
                                        "real_ssrf", max_count)
        print(f"  Loaded {len(data)} SSRF prompts from {len(payloads)} raw payloads")
        return data
    
    def load_path_traversal(self, max_count: int = 1000) -> List[Dict]:
        """Load path traversal payloads and convert to prompts."""
        print("Loading path traversal payloads...")
        payloads = []
        
        path_dir = self.payloads_dir / "Directory Traversal"
        if path_dir.exists():
            for md_file in path_dir.glob("*.md"):
                payloads.extend(self._extract_payloads_from_markdown(md_file))
            
            intruder_dir = path_dir / "Intruder"
            if intruder_dir.exists():
                for txt_file in intruder_dir.glob("*.txt"):
                    payloads.extend(self._extract_payloads_from_txt(txt_file))
        
        data = self._convert_to_prompts(payloads, self.path_templates,
                                        "real_path_traversal", max_count)
        print(f"  Loaded {len(data)} path traversal prompts from {len(payloads)} raw payloads")
        return data
    
    def load_xss(self, max_count: int = 1200) -> List[Dict]:
        """Load XSS payloads and convert to prompts."""
        print("Loading XSS payloads...")
        payloads = []
        
        xss_dir = self.payloads_dir / "XSS Injection"
        if xss_dir.exists():
            for md_file in xss_dir.glob("*.md"):
                payloads.extend(self._extract_payloads_from_markdown(md_file))
            
            # Intruder wordlists
            intruder_dir = xss_dir / "Intruders"
            if intruder_dir.exists():
                for txt_file in intruder_dir.glob("*.txt"):
                    payloads.extend(self._extract_payloads_from_txt(txt_file))
            
            # Files directory
            files_dir = xss_dir / "Files"
            if files_dir.exists():
                for txt_file in files_dir.glob("*.txt"):
                    payloads.extend(self._extract_payloads_from_txt(txt_file))
        
        data = self._convert_to_prompts(payloads, self.xss_templates,
                                        "real_xss", max_count)
        print(f"  Loaded {len(data)} XSS prompts from {len(payloads)} raw payloads")
        return data
    
    def load_prompt_injection(self, max_count: int = 500) -> List[Dict]:
        """Load prompt injection examples from PayloadsAllTheThings."""
        print("Loading prompt injection payloads...")
        payloads = []
        
        pi_dir = self.payloads_dir / "Prompt Injection"
        if pi_dir.exists():
            for md_file in pi_dir.glob("*.md"):
                payloads.extend(self._extract_payloads_from_markdown(md_file))
        
        # These are already in prompt format, minimal conversion needed
        data = []
        unique_payloads = list(set(payloads))[:max_count]
        for payload in unique_payloads:
            data.append({
                "text": payload,
                "label": 1,
                "source": "real_prompt_injection",
                "original_payload": payload
            })
        
        print(f"  Loaded {len(data)} prompt injection examples")
        return data
    
    def load_all(self) -> Tuple[List[Dict], Dict]:
        """Load all real payload categories and return combined dataset with stats."""
        all_data = []
        stats = {}
        
        # Load each category
        sql = self.load_sql_injection()
        all_data.extend(sql)
        stats["sql_injection"] = len(sql)
        
        cmd = self.load_command_injection()
        all_data.extend(cmd)
        stats["command_injection"] = len(cmd)
        
        ssrf = self.load_ssrf()
        all_data.extend(ssrf)
        stats["ssrf"] = len(ssrf)
        
        path = self.load_path_traversal()
        all_data.extend(path)
        stats["path_traversal"] = len(path)
        
        xss = self.load_xss()
        all_data.extend(xss)
        stats["xss"] = len(xss)
        
        pi = self.load_prompt_injection()
        all_data.extend(pi)
        stats["prompt_injection"] = len(pi)
        
        stats["total"] = len(all_data)
        
        print(f"\n=== Real Payload Summary ===")
        for cat, count in stats.items():
            print(f"  {cat}: {count}")
        
        return all_data, stats


class HuggingFaceSecurityDatasets:
    """Load real security datasets from HuggingFace."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_open_prompt_injection(self, max_count: int = 2000) -> List[Dict]:
        """
        Load guychuk/open-prompt-injection dataset.
        Contains 33,600 real indirect/context-based prompt injections.
        """
        print("Loading guychuk/open-prompt-injection dataset...")
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                "guychuk/open-prompt-injection",
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            data = []
            for split in dataset.keys():
                for item in dataset[split]:
                    # Extract text field (may vary by dataset structure)
                    text = None
                    for field in ['text', 'prompt', 'input', 'content', 'instruction']:
                        if field in item and item[field]:
                            text = str(item[field])
                            break
                    
                    if text and len(text) > 10:
                        data.append({
                            "text": text,
                            "label": 1,  # adversarial
                            "source": "real_open_prompt_injection"
                        })
                    
                    if len(data) >= max_count:
                        break
                if len(data) >= max_count:
                    break
            
            print(f"  Loaded {len(data)} indirect injection examples")
            return data
            
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            print("  Falling back to generated data...")
            return []
    
    def load_agent_harm(self, max_count: int = 1500) -> List[Dict]:
        """
        Load ai-safety-institute/AgentHarm dataset.
        Contains real tool misuse and harmful agent requests.
        """
        print("Loading ai-safety-institute/AgentHarm dataset...")
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                "ai-safety-institute/AgentHarm",
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            data = []
            for split in dataset.keys():
                for item in dataset[split]:
                    # Extract relevant fields
                    text = None
                    for field in ['prompt', 'text', 'input', 'request', 'task']:
                        if field in item and item[field]:
                            text = str(item[field])
                            break
                    
                    if text and len(text) > 10:
                        data.append({
                            "text": text,
                            "label": 1,  # adversarial (tool misuse)
                            "source": "real_agent_harm"
                        })
                    
                    if len(data) >= max_count:
                        break
                if len(data) >= max_count:
                    break
            
            print(f"  Loaded {len(data)} tool misuse examples")
            return data
            
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            print("  Falling back to generated data...")
            return []
    
    def load_mindgard_evaded(self, max_count: int = 1000) -> List[Dict]:
        """
        Load Mindgard/evaded-prompt-injection-and-jailbreak-samples.
        Contains evasion-enhanced injection attempts.
        """
        print("Loading Mindgard/evaded-prompt-injection dataset...")
        
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(
                "Mindgard/evaded-prompt-injection-and-jailbreak-samples",
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            data = []
            for split in dataset.keys():
                for item in dataset[split]:
                    text = None
                    for field in ['text', 'prompt', 'sample', 'input']:
                        if field in item and item[field]:
                            text = str(item[field])
                            break
                    
                    if text and len(text) > 10:
                        data.append({
                            "text": text,
                            "label": 1,
                            "source": "real_mindgard_evaded"
                        })
                    
                    if len(data) >= max_count:
                        break
                if len(data) >= max_count:
                    break
            
            print(f"  Loaded {len(data)} evaded injection examples")
            return data
            
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            return []
    
    def load_all(self) -> Tuple[List[Dict], Dict]:
        """Load all HuggingFace security datasets."""
        all_data = []
        stats = {}
        
        # Indirect/RAG injection
        opi = self.load_open_prompt_injection(max_count=2000)
        all_data.extend(opi)
        stats["open_prompt_injection"] = len(opi)
        
        # Tool misuse
        harm = self.load_agent_harm(max_count=1500)
        all_data.extend(harm)
        stats["agent_harm"] = len(harm)
        
        # Evaded injections
        evaded = self.load_mindgard_evaded(max_count=1000)
        all_data.extend(evaded)
        stats["mindgard_evaded"] = len(evaded)
        
        stats["total"] = len(all_data)
        
        print(f"\n=== HuggingFace Security Datasets Summary ===")
        for cat, count in stats.items():
            print(f"  {cat}: {count}")
        
        return all_data, stats


def save_real_dataset(data: List[Dict], output_path: Path):
    """Save the real payload dataset to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} examples to {output_path}")


if __name__ == "__main__":
    # Test the loaders
    print("Testing PayloadsAllTheThings loader...")
    loader = RealPayloadLoader()
    data1, stats1 = loader.load_all()
    
    print("\nTesting HuggingFace security datasets loader...")
    hf_loader = HuggingFaceSecurityDatasets()
    data2, stats2 = hf_loader.load_all()
    
    # Combine and save
    all_data = data1 + data2
    save_real_dataset(all_data, Path("data/real_payloads.json"))


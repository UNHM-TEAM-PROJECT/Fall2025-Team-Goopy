"""
Synthetic Q&A Generation Service
Converts catalog chunks into Q&A format to improve semantic matching with user queries.
Uses a larger, more capable LLM for question generation (only runs during indexing, not live).
"""
import re
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from transformers import pipeline
from config.settings import get_config, load_retrieval_config

class SyntheticQAGenerator:
    """Generates question-answer pairs from catalog text chunks using a larger LLM."""
    
    def __init__(self):
        # Ensure config is loaded
        load_retrieval_config()
        self.qa_pipeline = None
        self.output_file = None  # File handle for incremental saving
        
    def _get_pipeline(self):
        """
        Load a larger, more capable model for question generation.
        Since this only runs during index building (not live queries), we can afford it.
        """
        if self.qa_pipeline is None:
            import torch
            cfg = get_config()
            qa_cfg = cfg.get("synthetic_qa", {})
            
            # Get model from config
            model_name = qa_cfg.get("question_model", "google/flan-t5-large")
            # Check force_cpu setting
            force_cpu = qa_cfg.get("force_cpu", True)
            
            if force_cpu:
                device = -1
                print(f"Using CPU for question generation (force_cpu=true in config)")
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    print(f"   GPU available but disabled: {device_name}")
                    print(f"   Set force_cpu=false to try GPU")
            elif torch.cuda.is_available():
                device = 0
                device_name = torch.cuda.get_device_name(0)
                print(f"🚀 Using GPU: {device_name}")
            else:
                device = -1
                print(f"No GPU detected, using CPU")
            
            print(f"Loading question generation model: {model_name}")
            
            # Determine pipeline type based on model
            # T5/FLAN-T5/BART use text2text-generation
            # Phi-2, Qwen, Llama, Mistral use text-generation (causal LM)
            if any(name in model_name.lower() for name in ['t5', 'bart', 'pegasus']):
                pipeline_type = "text2text-generation"
            else:
                pipeline_type = "text-generation"
            
            print(f"  Using pipeline: {pipeline_type}")
            
            self.qa_pipeline = pipeline(
                pipeline_type,
                model=model_name,
                device=device,
                max_length=512,
                trust_remote_code=True  # Required for some models like Phi-2
            )
            self.pipeline_type = pipeline_type
            print(f"  ✓ Model loaded on {'GPU' if device == 0 else 'CPU'}")
        return self.qa_pipeline
    
    def generate_question_for_chunk(self, chunk_text: str, context_title: str = "") -> Optional[str]:
        """
        Use a larger LLM to generate ONE specific, natural question that this chunk answers.
        Returns None if chunk doesn't seem to contain question-answerable content.
        """
        # Skip very short chunks or list items
        if len(chunk_text.strip()) < 50:
            return None
        
        try:
            pipeline = self._get_pipeline()
            
            # Handle different output formats for different pipeline types
            if self.pipeline_type == "text2text-generation":
                # T5/FLAN-T5/BART: works best with clear, direct instructions
                prompt = (
                    f"Write a short, natural question that a graduate student would ask about this university policy. "
                    f"Keep it under 12 words.\n\n"
                    f"Policy: {chunk_text[:300]}\n\n"
                    "Question:"
                )
                result = pipeline(
                    prompt,
                    max_new_tokens=25,  # Reduced for brevity
                    temperature=0.7,  # Higher for variety
                    do_sample=True,
                    top_p=0.9,
                    num_return_sequences=1,
                    repetition_penalty=2.0  # Stronger penalty to reduce repetition
                )
                generated = result[0]["generated_text"].strip()
            else:
                # Causal LM (Phi-2, Qwen, Llama, Mistral): use proper instruction format
                tokenizer = pipeline.tokenizer
                
                # Phi-2 specific format: "Instruct: <task>\nOutput:"
                # This is critical - Phi-2 was trained with this exact format
                prompt = (
                    f"Instruct: Write one short question (under 12 words) that a graduate student would ask about this policy:\n\n"
                    f"{chunk_text[:300]}\n\n"
                    "Output:"
                )
                
                result = pipeline(
                    prompt,
                    max_new_tokens=20,  # Allow slightly more tokens for complete questions
                    temperature=0.2,  # Very low temperature to reduce hallucination
                    do_sample=True,
                    top_p=0.7,  # More focused sampling
                    num_return_sequences=1,
                    repetition_penalty=1.1,
                    return_full_text=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated = result[0]["generated_text"].strip()
            
            # Clean up the question - extract only up to first question mark
            question = generated
            
            # Stop at first question mark
            if '?' in question:
                question = question.split('?')[0] + '?'
            
            # Remove common prefixes
            question = re.sub(r'^Question:\s*', '', question, flags=re.IGNORECASE)
            question = re.sub(r'^Student question:\s*', '', question, flags=re.IGNORECASE)
            question = re.sub(r'^Answer:\s*', '', question, flags=re.IGNORECASE)
            question = re.sub(r'^Q:\s*', '', question, flags=re.IGNORECASE)
            question = re.sub(r'^A:\s*', '', question, flags=re.IGNORECASE)
            
            # Take only first line
            question = question.split('\n')[0].strip()
            
            # Clean up common artifacts
            question = re.sub(r'^\d+[\.\)]\s*', '', question)  # Remove numbering
            question = re.sub(r'^[-•*]\s*', '', question)  # Remove bullets
            question = re.sub(r'^["\']|["\']$', '', question)  # Remove quotes
            question = question.strip()
            
            # Ensure it ends with question mark
            if question and not question.endswith('?'):
                question += '?'
            
            # Validate minimum requirements
            if not question or len(question) < 10 or len(question) > 150:
                return None
            
            # Filter overly generic/formal questions
            lower_q = question.lower()
            
            # Reject meta-references to the text itself
            meta_terms = ['the text', 'this passage', 'above text', 'following text', 
                         'the document', 'this document', 'described above', 'mentioned above']
            if any(bad in lower_q for bad in meta_terms):
                return None
            
            # Reject overly generic starts (too formal)
            generic_starts = [
                'what is the process',
                'what are the requirements',
                'what is the policy',
                'what are the procedures',
                'what is the definition'
            ]
            if any(lower_q.startswith(bad) for bad in generic_starts):
                # Allow if it's specific enough (has program name, number, etc.)
                if not any(marker in lower_q for marker in ['master', 'ph.d', 'doctoral', 'certificate', 
                                                             'graduate', 'credit', 'gpa', 'grade']):
                    return None
            
            return question
            
        except Exception as e:
            print(f"Warning: Question generation failed for chunk: {e}")
            return None
    
    def create_qa_chunk(self, original_chunk: str, question: str, source_meta: Dict) -> Tuple[str, Dict]:
        """
        Create a Q&A formatted chunk that will semantically match user queries better.
        
        Format: "Question: {question}\n\nAnswer: {original_chunk}"
        """
        # Clean up the chunk - remove excessive whitespace
        clean_chunk = re.sub(r'\s+', ' ', original_chunk).strip()
        
        # Create Q&A format
        qa_text = f"Question: {question}\n\nAnswer: {clean_chunk}"
        
        # Copy metadata and mark as synthetic QA
        qa_meta = source_meta.copy()
        qa_meta['is_synthetic_qa'] = True
        qa_meta['original_question'] = question
        
        return qa_text, qa_meta
    
    
    def augment_chunks_with_qa(
        self, 
        chunks: List[Tuple[str, Dict]]
    ) -> List[Tuple[str, Dict]]:
        """
        For each chunk, generate ONE synthetic Q&A pair and add it to the index.
        Returns both original chunks AND synthetic Q&A chunks.
        Saves generated Q&A pairs incrementally to a JSONL file for monitoring.
        """
        augmented = []
        
        # Keep original chunks
        augmented.extend(chunks)
        
        # Get tier filter from config
        cfg = get_config()
        qa_cfg = cfg.get("synthetic_qa", {})
        tier_filter = qa_cfg.get("generate_for_tiers", "all")
        
        # Set up output file for incremental saving
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "synthetic_qa_generated.jsonl"
        
        # Open file in write mode (overwrites previous run)
        print(f"\nSaving generated Q&A pairs to: {output_path}")
        
        # Show tier filtering info
        if tier_filter == "all":
            print(f"Generating Q&A for ALL chunks (~11,000 chunks)")
            print(f"Estimated time: 6-15 hours on CPU")
        else:
            print(f"Generating Q&A for tiers: {tier_filter}")
            print(f"Estimated time: 1-3 hours on CPU (for ~2000 chunks)")
        
        print("(You can view this file in real-time as questions are generated)\n")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Add synthetic Q&A versions (ONE question per chunk)
            generated_count = 0
            skipped_count = 0
            filtered_count = 0
            duplicate_count = 0
            
            # Track generated questions to avoid duplicates
            seen_questions = set()
            
            for idx, (chunk_text, chunk_meta) in enumerate(chunks, 1):
                # Skip if already synthetic Q&A (avoid duplicates if run multiple times)
                if chunk_meta.get('is_synthetic_qa'):
                    continue
                
                # Tier filtering
                chunk_tier = chunk_meta.get('tier', 2)
                if tier_filter != "all" and chunk_tier not in tier_filter:
                    filtered_count += 1
                    continue
                
                context_title = chunk_meta.get('title', '')
                question = self.generate_question_for_chunk(chunk_text, context_title)
                
                if question:
                    # Check for duplicates (normalized comparison)
                    normalized_q = question.lower().strip()
                    if normalized_q in seen_questions:
                        duplicate_count += 1
                        if duplicate_count % 10 == 0:
                            print(f"  Skipped {duplicate_count} duplicates so far... (latest: {question[:50]}...)")
                        continue
                    
                    seen_questions.add(normalized_q)
                    
                    qa_chunk, qa_meta = self.create_qa_chunk(chunk_text, question, chunk_meta)
                    augmented.append((qa_chunk, qa_meta))
                    generated_count += 1
                    
                    # Save to file immediately (incremental)
                    record = {
                        "index": generated_count,
                        "original_chunk_index": idx,
                        "tier": chunk_tier,
                        "question": question,
                        "original_text": chunk_text,
                        "title": context_title,
                        "full_qa_chunk": qa_chunk
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    f.flush()  # Force write to disk immediately
                    
                    # Progress indicator
                    if generated_count % 10 == 0:
                        print(f"  Generated {generated_count} questions so far... (latest: {question[:60]}...)")
                else:
                    skipped_count += 1
        
        print(f"\n✓ Generated {generated_count} synthetic Q&A chunks from {len(chunks)} original chunks")
        if tier_filter != "all":
            print(f"  (Filtered {filtered_count} chunks - not in tiers {tier_filter})")
        print(f"  (Skipped {skipped_count} chunks - too short or unsuitable for Q&A)")
        print(f"  (Skipped {duplicate_count} duplicates - same question already generated)")
        print(f"  View results in: {output_path}\n")
        return augmented


# Global instance
_qa_generator = None

def get_qa_generator() -> SyntheticQAGenerator:
    global _qa_generator
    if _qa_generator is None:
        _qa_generator = SyntheticQAGenerator()
    return _qa_generator

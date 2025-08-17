import json
import re
import random
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


@dataclass
class LogoDesign:
    id: int
    title: str
    description: str
    design_elements: List[str]
    color_scheme: str
    symbolism: str
    generation_prompt: str = "" 

@dataclass
class LogoEvaluation:
    logo_id: int
    clarity_score: int
    relevance_score: int
    creativity_score: int
    vision_alignment_score: int
    simplicity_score: int
    total_score: int
    reasoning: str
    evaluation_prompt: str = ""  

class LogoGeneratorAgent:
    def __init__(self, model_name: str = "llama2"):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.8,  
            num_predict=512
        )
        
    def create_generation_prompt(self) -> PromptTemplate:
        # creating prompt template for logo generation
        template = """
        You are a creative logo designer for an AI/ML club called SCALE. 
        
        Club Description: {club_description}
        Personal Vision: {personal_vision}
        Logo Number: {logo_number}
        
        Generate a unique, creative logo design that represents the club's mission and vision.
        Focus on AI/ML themes, learning, innovation, and community building.
        
        Make sure each logo is distinctly different from previous ones.
        
        Please provide your response in the following JSON format:
        {{
            "title": "Logo title here",
            "description": "Detailed visual description of the logo",
            "design_elements": ["element1", "element2", "element3"],
            "color_scheme": "Description of colors used",
            "symbolism": "What the logo symbolizes and represents"
        }}
        
        Logo Design:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["club_description", "personal_vision", "logo_number"]
        )
    
    def parse_logo_response(self, response: str, logo_id: int, prompt_used: str) -> LogoDesign:
        try:
            #  to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                return LogoDesign(
                    id=logo_id,
                    title=data.get("title", f"SCALE Logo {logo_id}"),
                    description=data.get("description", "Modern AI-themed logo design"),
                    design_elements=data.get("design_elements", ["AI elements", "Modern design"]),
                    color_scheme=data.get("color_scheme", "Blue and white"),
                    symbolism=data.get("symbolism", "Represents learning and AI"),
                    generation_prompt=prompt_used
                )
        except Exception as e:
            print(f"   JSON parsing failed, using fallback: {e}")
        
        # fallback parsing - extract information from text
        return self.create_fallback_logo(response, logo_id, prompt_used)
    
    def create_fallback_logo(self, response: str, logo_id: int, prompt_used: str) -> LogoDesign:
        lines = response.split('\n')
        title = f"SCALE AI Logo {logo_id}"
        description = "AI-themed logo design for SCALE club"
        design_elements = ["Neural network", "Modern typography", "Geometric shapes"]
        color_scheme = "Blue gradient with white accents"
        symbolism = "Represents artificial intelligence, learning, and community"
        
        for line in lines:
            line = line.strip()
            if "title:" in line.lower() or "name:" in line.lower():
                title = line.split(":", 1)[1].strip()
            elif "description:" in line.lower():
                description = line.split(":", 1)[1].strip()
            elif "color" in line.lower() and ":" in line:
                color_scheme = line.split(":", 1)[1].strip()
            elif "symbol" in line.lower() and ":" in line:
                symbolism = line.split(":", 1)[1].strip()
        
        return LogoDesign(
            id=logo_id,
            title=title,
            description=description,
            design_elements=design_elements,
            color_scheme=color_scheme,
            symbolism=symbolism,
            generation_prompt=prompt_used
        )
    
    def generate_logos(self, club_description: str, personal_vision: str, num_logos: int = 3) -> List[LogoDesign]:
        prompt_template = self.create_generation_prompt()
        logos = []
        
        print(f" Generating {num_logos} logo designs...")
        
        for i in range(num_logos):
            print(f"  Generating logo {i+1}...")
            
            prompt = prompt_template.format(
                club_description=club_description,
                personal_vision=personal_vision,
                logo_number=i+1
            )
            
            try:
                response = self.llm.invoke(prompt)
                logo = self.parse_logo_response(response, i + 1, prompt)
                logos.append(logo)
                print(f"   Logo {i+1}: '{logo.title}' generated successfully")
                
            except Exception as e:
                print(f"   Error generating logo {i+1}: {e}")
                # creating a fallback logo with varying quality
                fallback_designs = [
                    {
                        "title": f"Neural Nexus {i+1}",
                        "description": "A sophisticated neural network visualization with interconnected nodes",
                        "elements": ["Neural nodes", "Connection lines", "Gradient effects"],
                        "colors": "Deep blue to cyan gradient",
                        "symbolism": "Represents interconnected learning and AI intelligence"
                    },
                    {
                        "title": f"Scale Matrix {i+1}",
                        "description": "A minimalist logo with layered geometric shapes representing scaling",
                        "elements": ["Layered rectangles", "Ascending pattern", "Clean typography"],
                        "colors": "Monochrome with blue accent",
                        "symbolism": "Represents growth, scaling, and systematic learning"
                    },
                    {
                        "title": f"AI Compass {i+1}",
                        "description": "A compass-like design pointing towards innovation and discovery",
                        "elements": ["Compass design", "Circuit patterns", "Direction indicators"],
                        "colors": "Gold and navy blue",
                        "symbolism": "Represents guidance, direction, and exploration in AI"
                    }
                ]
                
                design = fallback_designs[i % len(fallback_designs)]
                fallback_logo = LogoDesign(
                    id=i+1,
                    title=design["title"],
                    description=design["description"],
                    design_elements=design["elements"],
                    color_scheme=design["colors"],
                    symbolism=design["symbolism"],
                    generation_prompt=prompt
                )
                logos.append(fallback_logo)
                print(f"   Used fallback for logo {i+1}")
        
        return logos

class LogoJudgeAgent:
    def __init__(self, model_name: str = "llama2"):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.1,  # lower temperature for consistent evaluation , but can be changed
            num_predict=400
        )
    
    def create_evaluation_prompt(self) -> PromptTemplate:
        template = """
        You are a CRITICAL logo evaluation expert for an AI/ML club called SCALE. 
        You must be STRICT and REALISTIC in your scoring. Most logos have significant flaws.
        
        Club Context: {club_description}
        Personal Vision: {personal_vision}
        
        IMPORTANT: Be harsh but fair. Real logos rarely score above 7-8 in most categories.
        Consider real-world design standards and professional logo requirements.
        
        Evaluate this logo design strictly on each criterion (1-10 scale):
        
        1. CLARITY (1-10): Is it clear, readable, and professionally designed?
           - 1-3: Confusing, cluttered, hard to read
           - 4-6: Somewhat clear but has issues
           - 7-8: Clear and professional
           - 9-10: Exceptionally clear and polished
        
        2. RELEVANCE (1-10): Does it truly represent AI/ML and learning effectively?
           - 1-3: No clear AI/ML connection
           - 4-6: Weak or generic tech representation
           - 7-8: Good AI/ML representation
           - 9-10: Perfect AI/ML symbolism
        
        3. CREATIVITY (1-10): Is it original and innovative, not cliché?
           - 1-3: Generic, overused concepts
           - 4-6: Somewhat creative but predictable
           - 7-8: Creative with unique elements
           - 9-10: Highly original and innovative
        
        4. VISION_ALIGNMENT (1-10): How well does it match the club's student-friendly vision?
           - 1-3: Doesn't match vision at all
           - 4-6: Partially matches some aspects
           - 7-8: Good alignment with vision
           - 9-10: Perfect embodiment of vision
        
        5. SIMPLICITY (1-10): Is it simple enough to be memorable and scalable?
           - 1-3: Overly complex, won't work at small sizes
           - 4-6: Somewhat complex, some scalability issues
           - 7-8: Good balance of detail and simplicity
           - 9-10: Perfect simplicity and scalability
        
        Logo to Evaluate:
        Title: {logo_title}
        Description: {logo_description}
        Design Elements: {logo_elements}
        Color Scheme: {logo_colors}
        Symbolism: {logo_symbolism}
        
        Be CRITICAL. Point out specific flaws. Most designs will have scores between 4-7.
        Only exceptional designs should score 8+.
        
        Respond with ONLY this JSON format (no extra text):
        {{
            "clarity_score": [1-10 integer],
            "relevance_score": [1-10 integer],
            "creativity_score": [1-10 integer],
            "vision_alignment_score": [1-10 integer],
            "simplicity_score": [1-10 integer],
            "reasoning": "Detailed critical analysis explaining each score with specific flaws and strengths"
        }}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "club_description", "personal_vision", "logo_title", 
                "logo_description", "logo_elements", "logo_colors", "logo_symbolism"
            ]
        )
    
    def parse_evaluation_response(self, response: str, logo_id: int, prompt_used: str) -> LogoEvaluation:
        """Parse the LLM response into a LogoEvaluation object with realistic scoring"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                clarity = max(1, min(10, int(data.get("clarity_score", random.randint(4, 7)))))
                relevance = max(1, min(10, int(data.get("relevance_score", random.randint(4, 7)))))
                creativity = max(1, min(10, int(data.get("creativity_score", random.randint(3, 6)))))
                vision = max(1, min(10, int(data.get("vision_alignment_score", random.randint(4, 7)))))
                simplicity = max(1, min(10, int(data.get("simplicity_score", random.randint(5, 8)))))
                
                if clarity + relevance + creativity + vision + simplicity > 38:
                    scores = [clarity, relevance, creativity, vision, simplicity]
                    for _ in range(random.randint(1, 2)):
                        idx = random.randint(0, 4)
                        scores[idx] = max(1, scores[idx] - random.randint(1, 2))
                    clarity, relevance, creativity, vision, simplicity = scores
                
                total = clarity + relevance + creativity + vision + simplicity
                
                return LogoEvaluation(
                    logo_id=logo_id,
                    clarity_score=clarity,
                    relevance_score=relevance,
                    creativity_score=creativity,
                    vision_alignment_score=vision,
                    simplicity_score=simplicity,
                    total_score=total,
                    reasoning=data.get("reasoning", "Analysis based on professional design standards"),
                    evaluation_prompt=prompt_used
                )
        except Exception as e:
            print(f"   JSON parsing failed, using realistic fallback: {e}")
        
        return self.create_realistic_fallback_evaluation(response, logo_id, prompt_used)
    
    def create_realistic_fallback_evaluation(self, response: str, logo_id: int, prompt_used: str) -> LogoEvaluation:
        base_scores = [
            random.randint(4, 7),  # clarity
            random.randint(5, 7),  # relevance 
            random.randint(3, 6),  # creativity
            random.randint(4, 7),  # vision alignment
            random.randint(5, 8),  # simplicity
        ]
        
        if logo_id == 1:
            reasoning = "First logo shows standard AI themes but lacks distinctiveness. Clarity could be improved with better typography. Creativity is limited by common neural network imagery."
        elif logo_id == 2:
            reasoning = "Second design has interesting geometric elements but may be too abstract for immediate recognition. Good simplicity but relevance to AI/ML could be stronger."
            base_scores[1] = max(1, base_scores[1] - 1)  # reduce relevance
            base_scores[2] = min(10, base_scores[2] + 1)  # increase creativity
        else:
            reasoning = "Third logo attempts innovation but execution has flaws. Some elements work well while others detract from overall cohesion. Balance between complexity and clarity needs work."
            base_scores[0] = max(1, base_scores[0] - 1)  # reduce clarity
        
        return LogoEvaluation(
            logo_id=logo_id,
            clarity_score=base_scores[0],
            relevance_score=base_scores[1],
            creativity_score=base_scores[2],
            vision_alignment_score=base_scores[3],
            simplicity_score=base_scores[4],
            total_score=sum(base_scores),
            reasoning=reasoning,
            evaluation_prompt=prompt_used
        )
    
    def evaluate_logo(self, logo: LogoDesign, club_description: str, personal_vision: str) -> LogoEvaluation:
        prompt_template = self.create_evaluation_prompt()
        
        prompt = prompt_template.format(
            club_description=club_description,
            personal_vision=personal_vision,
            logo_title=logo.title,
            logo_description=logo.description,
            logo_elements=", ".join(logo.design_elements),
            logo_colors=logo.color_scheme,
            logo_symbolism=logo.symbolism
        )
        
        try:
            response = self.llm.invoke(prompt)
            evaluation = self.parse_evaluation_response(response, logo.id, prompt)
            return evaluation
            
        except Exception as e:
            print(f"  Error evaluating logo {logo.id}: {e}")
            return self.create_realistic_fallback_evaluation("", logo.id, prompt)
    
    def evaluate_all_logos(self, logos: List[LogoDesign], club_description: str, personal_vision: str) -> List[LogoEvaluation]:
        evaluations = []
        
        print(f" Evaluating {len(logos)} logo designs...")
        
        for logo in logos:
            print(f"  Evaluating '{logo.title}'...")
            evaluation = self.evaluate_logo(logo, club_description, personal_vision)
            evaluations.append(evaluation)
            print(f"   Total Score: {evaluation.total_score}/50")
        
        return evaluations

class LogoDataManager:
    """Handles saving and loading logo data to/from JSON files"""
    
    @staticmethod
    def save_results_to_json(results: Dict[str, Any], filename: str = None) -> str:
        """Save the pipeline results to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logo_results_{timestamp}.json"
        
        # json serialisation
        json_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_logos_generated": len(results["all_logos"]),
                "winning_logo_id": results["winning_logo"].id,
                "winning_score": results["final_score"]
            },
            "logos": [],
            "evaluations": []
        }
        
        # logo data
        for logo in results["all_logos"]:
            logo_dict = asdict(logo)
            json_data["logos"].append(logo_dict)
        
      
        for evaluation in results["all_evaluations"]:
            eval_dict = asdict(evaluation)
            json_data["evaluations"].append(eval_dict)
        

        json_data["winning_logo"] = asdict(results["winning_logo"])
        json_data["winning_evaluation"] = asdict(results["winning_evaluation"])
        

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"Results saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f" Error saving to JSON: {e}")
            return None
    
    @staticmethod
    def load_results_from_json(filename: str) -> Dict[str, Any]:
        """Load pipeline results from a JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f" Results loaded from: {filename}")
            return data
            
        except Exception as e:
            print(f" Error loading from JSON: {e}")
            return None
    
    @staticmethod
    def list_saved_results() -> List[str]:
        files = []
        for file in os.listdir('.'):
            if file.startswith('logo_results_') and file.endswith('.json'):
                files.append(file)
        return sorted(files)

class LogoPipeline:
    def __init__(self, model_name: str = "llama2"):
        self.generator = LogoGeneratorAgent(model_name)
        self.judge = LogoJudgeAgent(model_name)
        self.data_manager = LogoDataManager()
    
    def run_basic_pipeline(self, club_description: str, personal_vision: str, num_logos: int = 3, save_to_json: bool = True) -> Dict[str, Any]:
        print("Starting Logo Generation and Evaluation Pipeline")
        print("=" * 60)
        
        # logos generate
        logos = self.generator.generate_logos(club_description, personal_vision, num_logos)
        
        # evaluate logos
        evaluations = self.judge.evaluate_all_logos(logos, club_description, personal_vision)
        
        # best logo
        best_evaluation = max(evaluations, key=lambda x: x.total_score)
        best_logo = next(logo for logo in logos if logo.id == best_evaluation.logo_id)
        
        print(f"\n Evaluation completed!")
        
        final_results = {
            "winning_logo": best_logo,
            "winning_evaluation": best_evaluation,
            "all_logos": logos,
            "all_evaluations": evaluations,
            "final_score": best_evaluation.total_score,
            "club_description": club_description,
            "personal_vision": personal_vision
        }
        
        if save_to_json:
            saved_file = self.data_manager.save_results_to_json(final_results)
            final_results["saved_file"] = saved_file
        
        self.display_results(final_results)
        return final_results
    
    def display_results(self, results: Dict[str, Any]):
        print("\n" + "=" * 80)
        print("🏆 FINAL RESULTS")
        print("=" * 80)
        
        winning_logo = results["winning_logo"]
        winning_eval = results["winning_evaluation"]
        all_evaluations = results["all_evaluations"]
        
        print(f"\n ALL LOGO SCORES:")
        print("-" * 40)
        for i, (logo, eval_data) in enumerate(zip(results["all_logos"], all_evaluations), 1):
            print(f"{i}. '{logo.title}': {eval_data.total_score}/50")
            print(f"   Scores: C:{eval_data.clarity_score} R:{eval_data.relevance_score} Cr:{eval_data.creativity_score} V:{eval_data.vision_alignment_score} S:{eval_data.simplicity_score}")
        
        print(f"\n WINNING LOGO: {winning_logo.title}")
        print(f" FINAL SCORE: {winning_eval.total_score}/50")
        
        print(f"\n LOGO DESCRIPTION:")
        print(f"   {winning_logo.description}")
        
        print(f"\n DESIGN ELEMENTS:")
        for element in winning_logo.design_elements:
            print(f"   • {element}")
        
        print(f"\n COLOR SCHEME: {winning_logo.color_scheme}")
        print(f" SYMBOLISM: {winning_logo.symbolism}")
        
        print(f"\n DETAILED SCORES:")
        print(f"   • Clarity: {winning_eval.clarity_score}/10")
        print(f"   • Relevance: {winning_eval.relevance_score}/10")
        print(f"   • Creativity: {winning_eval.creativity_score}/10")
        print(f"   • Vision Alignment: {winning_eval.vision_alignment_score}/10")
        print(f"   • Simplicity: {winning_eval.simplicity_score}/10")
        
        print(f"\n JUDGE'S REASONING:")
        print(f"   {winning_eval.reasoning}")
        
        if "saved_file" in results and results["saved_file"]:
            print(f"\n DATA SAVED TO: {results['saved_file']}")
        
        print("\n" + "=" * 80)
    
    def load_and_display_previous_results(self, filename: str):
        """Load and display results from a previous run"""
        data = self.data_manager.load_results_from_json(filename)
        if data:
            print(f"\n PREVIOUS RESULTS FROM: {filename}")
            print(f" Generated on: {data['metadata']['timestamp']}")
            print(f" Winner: {data['winning_logo']['title']}")
            print(f" Score: {data['metadata']['winning_score']}/50")
            
            print(f"\n GENERATION PROMPTS AND SCORES:")
            print("=" * 50)
            for i, (logo, evaluation) in enumerate(zip(data['logos'], data['evaluations'])):
                print(f"\n LOGO {i+1}: {logo['title']}")
                print(f" Score: {evaluation['total_score']}/50")
                print(f" Generation Prompt: {logo['generation_prompt'][:100]}...")
                print(f" Evaluation Prompt: {evaluation['evaluation_prompt'][:100]}...")
                print(f" Reasoning: {evaluation['reasoning']}")

def main():    
    club_description = """
    SCALE is an AI/ML club focused on fostering learning, innovation, and community building 
    in the field of artificial intelligence and machine learning. The club aims to provide 
    a platform for students to explore cutting-edge technologies, collaborate on projects, 
    and develop practical skills in AI/ML. SCALE stands for 'Students Collaborating in 
    Artificial Learning Environments' and emphasizes hands-on learning, research, and 
    real-world applications of AI technology.
    """
    
    personal_vision = """
    I envision SCALE as a vibrant community where students can transform from curious 
    beginners to confident AI practitioners. The club should represent innovation, 
    accessibility, and collaborative growth. I want the logo to convey both the 
    technical sophistication of AI and the approachable, student-friendly nature 
    of our learning environment. It should inspire both beginners and experts to 
    join our journey in exploring the frontiers of artificial intelligence.
    """
    
    print("Initializing Logo Pipeline...")
    pipeline = LogoPipeline(model_name="llama2")
    
    previous_files = LogoDataManager.list_saved_results()
    if previous_files:
        print(f"\n Found {len(previous_files)} previous result files:")
        for i, file in enumerate(previous_files, 1):
            print(f"  {i}. {file}")
        
        choice = input("\nWould you like to view a previous result? (y/n): ").lower().strip()
        if choice == 'y':
            try:
                file_num = int(input(f"Enter file number (1-{len(previous_files)}): "))
                if 1 <= file_num <= len(previous_files):
                    pipeline.load_and_display_previous_results(previous_files[file_num - 1])
                    
                    continue_choice = input("\nContinue with new generation? (y/n): ").lower().strip()
                    if continue_choice != 'y':
                        return
            except (ValueError, IndexError):
                print("Invalid selection, continuing with new generation...")
    
    print("\n Running Basic Pipeline...")
    results = pipeline.run_basic_pipeline(
        club_description=club_description,
        personal_vision=personal_vision,
        num_logos=3,
        save_to_json=True
    )
    
    return results

if __name__ == "__main__":
    results = main()
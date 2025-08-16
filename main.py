import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Simple data classes to avoid Pydantic issues
@dataclass
class LogoDesign:
    """Simple logo design data class"""
    id: int
    title: str
    description: str
    design_elements: List[str]
    color_scheme: str
    symbolism: str

@dataclass
class LogoEvaluation:
    """Simple logo evaluation data class"""
    logo_id: int
    clarity_score: int
    relevance_score: int
    creativity_score: int
    vision_alignment_score: int
    simplicity_score: int
    total_score: int
    reasoning: str

class LogoGeneratorAgent:
    """Agent responsible for generating logo designs"""
    
    def __init__(self, model_name: str = "llama2"):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.8,  # Higher creativity
            num_predict=512
        )
        
    def create_generation_prompt(self) -> PromptTemplate:
        """Create prompt template for logo generation"""
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
    
    def parse_logo_response(self, response: str, logo_id: int) -> LogoDesign:
        """Parse the LLM response into a LogoDesign object"""
        try:
            # Try to extract JSON from response
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
                    symbolism=data.get("symbolism", "Represents learning and AI")
                )
        except Exception as e:
            print(f"    ⚠️ JSON parsing failed, using fallback: {e}")
        
        # Fallback parsing - extract information from text
        return self.create_fallback_logo(response, logo_id)
    
    def create_fallback_logo(self, response: str, logo_id: int) -> LogoDesign:
        """Create a logo design from unstructured text response"""
        lines = response.split('\n')
        title = f"SCALE AI Logo {logo_id}"
        description = "AI-themed logo design for SCALE club"
        design_elements = ["Neural network", "Modern typography", "Geometric shapes"]
        color_scheme = "Blue gradient with white accents"
        symbolism = "Represents artificial intelligence, learning, and community"
        
        # Try to extract information from the response
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
            symbolism=symbolism
        )
    
    def generate_logos(self, club_description: str, personal_vision: str, num_logos: int = 3) -> List[LogoDesign]:
        """Generate multiple logo designs"""
        prompt_template = self.create_generation_prompt()
        logos = []
        
        print(f"🎨 Generating {num_logos} logo designs...")
        
        for i in range(num_logos):
            print(f"  Generating logo {i+1}...")
            
            prompt = prompt_template.format(
                club_description=club_description,
                personal_vision=personal_vision,
                logo_number=i+1
            )
            
            try:
                response = self.llm.invoke(prompt)
                logo = self.parse_logo_response(response, i + 1)
                logos.append(logo)
                print(f"  ✅ Logo {i+1}: '{logo.title}' generated successfully")
                
            except Exception as e:
                print(f"  ❌ Error generating logo {i+1}: {e}")
                # Create a fallback logo
                fallback_logo = LogoDesign(
                    id=i+1,
                    title=f"SCALE AI Logo {i+1}",
                    description=f"A modern logo design for SCALE AI club featuring AI-themed elements",
                    design_elements=["Neural network", "Geometric shapes", "Modern typography"],
                    color_scheme="Blue and white gradient",
                    symbolism="Represents learning, growth, and artificial intelligence"
                )
                logos.append(fallback_logo)
                print(f"  ⚠️ Used fallback for logo {i+1}")
        
        return logos

class LogoJudgeAgent:
    """Agent responsible for evaluating and judging logos"""
    
    def __init__(self, model_name: str = "llama2"):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.3,  # Lower temperature for consistent evaluation
            num_predict=512
        )
    
    def create_evaluation_prompt(self) -> PromptTemplate:
        """Create prompt template for logo evaluation"""
        template = """
        You are an expert logo evaluator for an AI/ML club called SCALE.
        
        Club Context: {club_description}
        Personal Vision: {personal_vision}
        
        Evaluate the following logo design based on these criteria (score 1-10 for each):
        
        1. CLARITY (1-10): How clear and readable is the design?
        2. RELEVANCE (1-10): How well does it represent AI/ML and learning?
        3. CREATIVITY (1-10): How original and innovative is the design?
        4. VISION_ALIGNMENT (1-10): How well does it align with the club's vision?
        5. SIMPLICITY (1-10): How simple yet effective is the design?
        
        Logo to Evaluate:
        Title: {logo_title}
        Description: {logo_description}
        Design Elements: {logo_elements}
        Color Scheme: {logo_colors}
        Symbolism: {logo_symbolism}
        
        Please provide your evaluation in the following JSON format:
        {{
            "clarity_score": 8,
            "relevance_score": 9,
            "creativity_score": 7,
            "vision_alignment_score": 8,
            "simplicity_score": 8,
            "total_score": 40,
            "reasoning": "Detailed explanation of your scoring decisions"
        }}
        
        Evaluation:
        """
        
        return PromptTemplate(
            template=template,
            input_variables=[
                "club_description", "personal_vision", "logo_title", 
                "logo_description", "logo_elements", "logo_colors", "logo_symbolism"
            ]
        )
    
    def parse_evaluation_response(self, response: str, logo_id: int) -> LogoEvaluation:
        """Parse the LLM response into a LogoEvaluation object"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                clarity = int(data.get("clarity_score", 7))
                relevance = int(data.get("relevance_score", 7))
                creativity = int(data.get("creativity_score", 7))
                vision = int(data.get("vision_alignment_score", 7))
                simplicity = int(data.get("simplicity_score", 7))
                total = clarity + relevance + creativity + vision + simplicity
                
                return LogoEvaluation(
                    logo_id=logo_id,
                    clarity_score=clarity,
                    relevance_score=relevance,
                    creativity_score=creativity,
                    vision_alignment_score=vision,
                    simplicity_score=simplicity,
                    total_score=total,
                    reasoning=data.get("reasoning", "Logo shows good design principles")
                )
        except Exception as e:
            print(f"    ⚠️ JSON parsing failed, using fallback evaluation: {e}")
        
        # Fallback evaluation
        return self.create_fallback_evaluation(response, logo_id)
    
    def create_fallback_evaluation(self, response: str, logo_id: int) -> LogoEvaluation:
        """Create a fallback evaluation from unstructured text"""
        # Extract numbers from response for scoring
        numbers = re.findall(r'\b([1-9]|10)\b', response)
        
        if len(numbers) >= 5:
            scores = [int(num) for num in numbers[:5]]
        else:
            scores = [7, 7, 7, 7, 7]  # Default moderate scores
        
        return LogoEvaluation(
            logo_id=logo_id,
            clarity_score=scores[0],
            relevance_score=scores[1],
            creativity_score=scores[2],
            vision_alignment_score=scores[3],
            simplicity_score=scores[4],
            total_score=sum(scores),
            reasoning=f"Evaluation based on design quality and AI relevance. Response: {response[:200]}..."
        )
    
    def evaluate_logo(self, logo: LogoDesign, club_description: str, personal_vision: str) -> LogoEvaluation:
        """Evaluate a single logo design"""
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
            evaluation = self.parse_evaluation_response(response, logo.id)
            return evaluation
            
        except Exception as e:
            print(f"  ❌ Error evaluating logo {logo.id}: {e}")
            # Fallback evaluation
            return LogoEvaluation(
                logo_id=logo.id,
                clarity_score=7,
                relevance_score=7,
                creativity_score=7,
                vision_alignment_score=7,
                simplicity_score=7,
                total_score=35,
                reasoning="Fallback evaluation due to processing error"
            )
    
    def evaluate_all_logos(self, logos: List[LogoDesign], club_description: str, personal_vision: str) -> List[LogoEvaluation]:
        """Evaluate all logo designs"""
        evaluations = []
        
        print(f"🏆 Evaluating {len(logos)} logo designs...")
        
        for logo in logos:
            print(f"  Evaluating '{logo.title}'...")
            evaluation = self.evaluate_logo(logo, club_description, personal_vision)
            evaluations.append(evaluation)
            print(f"  ✅ Total Score: {evaluation.total_score}/50")
        
        return evaluations

class LogoPipeline:
    """Main pipeline orchestrating the logo generation and evaluation process"""
    
    def __init__(self, model_name: str = "llama2"):
        self.generator = LogoGeneratorAgent(model_name)
        self.judge = LogoJudgeAgent(model_name)
        self.threshold_score = 40  # Minimum score for acceptance
        self.max_iterations = 3    # Maximum iterations for improvement
    
    def run_pipeline(self, club_description: str, personal_vision: str, num_logos: int = 3, iterative: bool = False) -> Dict[str, Any]:
        """Run the complete logo generation and evaluation pipeline"""
        
        print("🚀 Starting Logo Generation and Evaluation Pipeline")
        print("=" * 60)
        
        iteration = 1
        best_score = 0
        best_logo = None
        best_evaluation = None
        all_results = []
        
        while iteration <= self.max_iterations:
            print(f"\n📍 ITERATION {iteration}")
            print("-" * 40)
            
            # Generate logos
            logos = self.generator.generate_logos(club_description, personal_vision, num_logos)
            
            # Evaluate logos
            evaluations = self.judge.evaluate_all_logos(logos, club_description, personal_vision)
            
            # Find best logo in current iteration
            current_best_eval = max(evaluations, key=lambda x: x.total_score)
            current_best_logo = next(logo for logo in logos if logo.id == current_best_eval.logo_id)
            
            print(f"\n🏅 Best logo this iteration: '{current_best_logo.title}' (Score: {current_best_eval.total_score}/50)")
            
            # Store results
            iteration_result = {
                "iteration": iteration,
                "logos": logos,
                "evaluations": evaluations,
                "best_logo": current_best_logo,
                "best_evaluation": current_best_eval
            }
            all_results.append(iteration_result)
            
            # Update overall best
            if current_best_eval.total_score > best_score:
                best_score = current_best_eval.total_score
                best_logo = current_best_logo
                best_evaluation = current_best_eval
            
            # Check stopping conditions
            if not iterative:
                break
                
            if best_score >= self.threshold_score:
                print(f"🎉 Threshold reached! Best score: {best_score}/50")
                break
                
            if iteration >= self.max_iterations:
                print(f"🔄 Max iterations reached. Best score: {best_score}/50")
                break
                
            iteration += 1
            print(f"\n🔄 Score {current_best_eval.total_score} below threshold {self.threshold_score}. Continuing...")
        
        # Prepare final results
        final_results = {
            "winning_logo": best_logo,
            "winning_evaluation": best_evaluation,
            "total_iterations": iteration - 1 if not iterative else iteration,
            "all_iterations": all_results,
            "final_score": best_score
        }
        
        self.display_results(final_results)
        return final_results
    
    def display_results(self, results: Dict[str, Any]):
        """Display the final results in a clean format"""
        print("\n" + "=" * 80)
        print("🏆 FINAL RESULTS")
        print("=" * 80)
        
        winning_logo = results["winning_logo"]
        winning_eval = results["winning_evaluation"]
        
        print(f"\n🎨 WINNING LOGO: {winning_logo.title}")
        print(f"📊 FINAL SCORE: {winning_eval.total_score}/50")
        print(f"🔄 ITERATIONS: {results['total_iterations']}")
        
        print(f"\n📝 LOGO DESCRIPTION:")
        print(f"   {winning_logo.description}")
        
        print(f"\n🎯 DESIGN ELEMENTS:")
        for element in winning_logo.design_elements:
            print(f"   • {element}")
        
        print(f"\n🎨 COLOR SCHEME: {winning_logo.color_scheme}")
        print(f"🔮 SYMBOLISM: {winning_logo.symbolism}")
        
        print(f"\n📊 DETAILED SCORES:")
        print(f"   • Clarity: {winning_eval.clarity_score}/10")
        print(f"   • Relevance: {winning_eval.relevance_score}/10")
        print(f"   • Creativity: {winning_eval.creativity_score}/10")
        print(f"   • Vision Alignment: {winning_eval.vision_alignment_score}/10")
        print(f"   • Simplicity: {winning_eval.simplicity_score}/10")
        
        print(f"\n💭 JUDGE'S REASONING:")
        print(f"   {winning_eval.reasoning}")
        
        print("\n" + "=" * 80)

    def display_detailed_summary(self, results: Dict[str, Any]):
        """Display detailed summary of all iterations"""
        print("\n" + "🔍 DETAILED ITERATION SUMMARY")
        print("=" * 60)
        
        for iteration_data in results["all_iterations"]:
            iteration = iteration_data["iteration"]
            print(f"\nIteration {iteration}:")
            
            for i, (logo, eval_data) in enumerate(zip(iteration_data["logos"], iteration_data["evaluations"])):
                print(f"  Logo {i+1}: '{logo.title}' - Score: {eval_data.total_score}/50")

# Example usage and main execution
def main():
    """Main function to run the logo pipeline"""
    
    # SCALE club description and personal vision
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
    
    # Initialize and run pipeline
    print("Initializing Logo Pipeline...")
    pipeline = LogoPipeline(model_name="llama2")
    
    # Run basic pipeline (3 logos, single iteration)
    print("\n🎯 Running Basic Pipeline...")
    basic_results = pipeline.run_pipeline(
        club_description=club_description,
        personal_vision=personal_vision,
        num_logos=3,
        iterative=False
    )
    
    # Show detailed summary
    pipeline.display_detailed_summary(basic_results)
    
    # Run iterative pipeline (bonus feature)
    print("\n\n🎯 Running Iterative Pipeline...")
    iterative_results = pipeline.run_pipeline(
        club_description=club_description,
        personal_vision=personal_vision,
        num_logos=3,
        iterative=True
    )
    
    # Show detailed summary
    pipeline.display_detailed_summary(iterative_results)
    
    return iterative_results

if __name__ == "__main__":
    results = main()
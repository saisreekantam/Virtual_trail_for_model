from crewai import Agent, Task, Crew, Process
import os
import cv2
import numpy as np
import torch
from PIL import Image
import subprocess
import shutil
from typing import Dict, List, Any

class VITONHD_Agent:
    def __init__(self, viton_dir: str = "./VITON-HD"):
        """
        Initialize the VITON-HD agent with the path to the VITON-HD directory
        
        Args:
            viton_dir: Path to the VITON-HD directory
        """
        self.viton_dir = viton_dir
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            os.path.join(self.viton_dir, "datasets/test/cloth"),
            os.path.join(self.viton_dir, "datasets/test/cloth-mask"),
            os.path.join(self.viton_dir, "results")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def create_cloth_mask(self, cloth_image_path: str) -> str:
        """
        Create a binary cloth mask from the input cloth image
        
        Args:
            cloth_image_path: Path to the cloth image
            
        Returns:
            Path to the generated mask
        """
        # Extract filename without extension
        filename = os.path.basename(cloth_image_path)
        name, ext = os.path.splitext(filename)
        
        # Read the image
        img = cv2.imread(cloth_image_path)
        if img is None:
            raise ValueError(f"Could not read image at {cloth_image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to create binary mask
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Create black background
        h, w = mask.shape
        black_bg = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Set mask region to white
        black_bg[mask > 0] = (255, 255, 255)
        
        # Save paths
        cloth_save_path = os.path.join(self.viton_dir, "datasets/test/cloth", filename)
        mask_save_path = os.path.join(self.viton_dir, "datasets/test/cloth-mask", filename)
        
        # Copy original cloth to the cloth directory
        shutil.copy(cloth_image_path, cloth_save_path)
        
        # Save the mask
        cv2.imwrite(mask_save_path, black_bg)
        
        print(f"Created cloth mask at {mask_save_path}")
        return mask_save_path
    
    def create_test_pairs(self, model_image_id: str, cloth_image_name: str) -> str:
        """
        Create test_pairs.txt file with model and cloth pair
        
        Args:
            model_image_id: ID of the model image (e.g. '000010_0')
            cloth_image_name: Filename of the cloth image
            
        Returns:
            Path to the test_pairs.txt file
        """
        # Remove extension if present
        cloth_name = os.path.splitext(cloth_image_name)[0]
        
        # Create test_pairs.txt content
        content = f"{model_image_id} {cloth_name}\n"
        
        # Save path
        pairs_path = os.path.join(self.viton_dir, "datasets/test_pairs.txt")
        
        # Write to file
        with open(pairs_path, 'w') as f:
            f.write(content)
            
        print(f"Created test pairs file at {pairs_path}")
        return pairs_path
    
    def run_viton_hd(self, output_name: str = "test_output") -> str:
        """
        Run the VITON-HD model with the prepared inputs
        
        Args:
            output_name: Name for the output directory
            
        Returns:
            Path to the result image
        """
        # Change to VITON-HD directory
        orig_dir = os.getcwd()
        os.chdir(self.viton_dir)
        
        try:
            # Run the test script
            cmd = ["python", "test.py", "--name", output_name]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"Error running VITON-HD: {stderr.decode()}")
                raise Exception(f"VITON-HD process failed with exit code {process.returncode}")
                
            # Get the result image path (assuming first pair in test_pairs.txt)
            with open("datasets/test_pairs.txt", 'r') as f:
                first_line = f.readline().strip()
                model_id, cloth_name = first_line.split()
                
            result_name = f"{model_id.split('_')[0]}_{cloth_name}"
            result_path = os.path.join("results", output_name, f"{result_name}.jpg")
            
            if not os.path.exists(result_path):
                raise FileNotFoundError(f"Result image not found at {result_path}")
                
            absolute_result_path = os.path.abspath(result_path)
            print(f"Generated result image at {absolute_result_path}")
            
            return absolute_result_path
            
        finally:
            # Change back to original directory
            os.chdir(orig_dir)
            
    def process_cloth_image(self, cloth_image_path: str, model_image_id: str, output_name: str = "test_output") -> str:
        """
        Process a cloth image through the entire VITON-HD pipeline
        
        Args:
            cloth_image_path: Path to the cloth image
            model_image_id: ID of the model image
            output_name: Name for the output directory
            
        Returns:
            Path to the result image
        """
        # 1. Create cloth mask
        self.create_cloth_mask(cloth_image_path)
        
        # 2. Create test_pairs.txt
        cloth_filename = os.path.basename(cloth_image_path)
        self.create_test_pairs(model_image_id, cloth_filename)
        
        # 3. Run VITON-HD
        result_path = self.run_viton_hd(output_name)
        
        return result_path

# CrewAI Integration
class VITONHDCrew:
    def __init__(self, viton_dir: str = "./VITON-HD"):
        self.viton_agent = VITONHD_Agent(viton_dir)
        
        # Create the agent
        self.cloth_processor = Agent(
            role="Virtual Try-On Specialist",
            goal="Process clothing images and create virtual try-on results",
            backstory="I am an AI agent specializing in virtual try-on technology, "
                    "helping users visualize how clothes would look on models.",
            verbose=True,
            allow_delegation=False
        )
        
    def create_cloth_processing_task(self, cloth_image_path: str, model_image_id: str) -> Task:
        """Create a task for processing a cloth image"""
        return Task(
            description=f"Process the cloth image at {cloth_image_path} with model {model_image_id}",
            agent=self.cloth_processor,
            expected_output="Path to the resulting try-on image",
            context={
                "cloth_image_path": cloth_image_path,
                "model_image_id": model_image_id,
                "viton_agent": self.viton_agent  # Pass the VITON-HD agent for use in task execution
            }
        )
        
    def execute_cloth_processing(self, cloth_image_path: str, model_image_id: str) -> str:
        """Execute the cloth processing task"""
        task = self.create_cloth_processing_task(cloth_image_path, model_image_id)
        
        crew = Crew(
            agents=[self.cloth_processor],
            tasks=[task],
            verbose=2,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        return result

# Example usage
if __name__ == "__main__":
    # Create the VITON-HD crew
    viton_crew = VITONHDCrew(viton_dir="./VITON-HD")
    
    # Process a cloth image
    result_path = viton_crew.execute_cloth_processing(
        cloth_image_path="path/to/your/cloth_image.jpg",
        model_image_id="000010_0"  # ID of a model image in the VITON-HD dataset
    )
    
    print(f"Final result image: {result_path}")

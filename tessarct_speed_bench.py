import time
from classes.pdf_parser import JoradpFileParse
from classes.ocr_processor import OcrProcessor
from classes.image_builder import ImageBuilder
from classes.joradp_importer import JoradpImporter

# Let's define a function that runs a single version of the processing
def run_processing(use_parallel=False):
    # Record the start time
    start_time = time.time()
    
    # Print which version we're running
    print(f"\nRunning {'parallel' if use_parallel else 'sequential'} version...")
    

    
    # Choose the appropriate rotation method
    if use_parallel:
        parserImages = JoradpFileParse("./data_test/F2024080.pdf")
        ocr = OcrProcessor()
        parserImages.get_images_with_pymupdf()
        parserImages.resize_image_to_fit_ocr()
        parserImages.crop_all_images(top=120, left=80, right=80, bottom=100)
        parserImages.adjust_all_images_rotations_parallel()
        parserImages.parse_images_to_text_structure(ocr)
        
    else:
        parserImages = JoradpFileParse("./data_test/F2024080.pdf")
        ocr = OcrProcessor()
        parserImages.get_images()
        parserImages.resize_image_to_fit_ocr()
        parserImages.crop_all_images(top=120, left=80, right=80, bottom=100)
        parserImages.adjust_all_images_rotations()
        parserImages.parse_images_to_text_structure_optimized(ocr)
    
    # Calculate and return the elapsed time
    elapsed_time = time.time() - start_time
    return elapsed_time

# Run both versions and measure their times
print("Starting performance comparison...")

# Run sequential version
sequential_time = run_processing(use_parallel=False)
print(f"Sequential version took: {sequential_time:.2f} seconds")

# Add a small pause between runs to let system resources settle
time.sleep(2)

# Run parallel version
parallel_time = run_processing(use_parallel=True)
print(f"Parallel version took: {parallel_time:.2f} seconds")

# Calculate and display the speedup
speedup = sequential_time / parallel_time
print(f"\nPerformance Summary:")
print(f"Sequential time: {sequential_time:.2f} seconds")
print(f"Parallel time:   {parallel_time:.2f} seconds")
print(f"Speedup factor:  {speedup:.2f}x")

if speedup > 1:
    print(f"The parallel version was {((speedup - 1) * 100):.1f}% faster")
else:
    print(f"The sequential version was {((1/speedup - 1) * 100):.1f}% faster")
##################################################
##  SCRIPT TO EXTRACT DATA FROM FOLDER OF       ##
## USCG MARINERS REPORT PDFS BASED ON KEYWORDS  ##
##################################################

# IMPORT LIBRARIES
import fitz  # PyMuPDF
import os
from pathlib import Path

# FUNCTION TO SEARCH FOR KEYWORDS AND COMBINE RESULTS INTO A PDF
def extract_multi_keyword_pages(folder_path, keywords, output_filename):
    result_pdf = fitz.open()
    # Clean keywords: lowercase and remove extra whitespace
    keywords = [k.lower().strip() for k in keywords]
    
    found_any = False

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            
            try:
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text().lower()
                    
                    # Check if the keywords are in the text
                    if any(word in text for word in keywords):
                        result_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                        found_any = True
                doc.close()
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if found_any:
        result_pdf.save(output_filename)
        print(f"Success! Compiled into {output_filename}")
    else:
        print("No matches found for any keywords.")
    
    result_pdf.close()

# Usage
home_dir = Path.home()
# Change file path for pdfs of interest
file_path = home_dir / "Documents" / "AIS" / "USCG_LNM" / "District1" / "D01LNM2022"
# Change keywords depending on search
my_keywords = ["wind", "southcoast", "south fork", "sunrise", "beacon", "vineyard", "empire", "bay state", "geotechnical"]
extract_multi_keyword_pages(file_path, my_keywords, file_path / "Combined_Results.pdf")
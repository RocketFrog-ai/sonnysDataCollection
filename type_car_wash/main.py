import sys
from type_car_wash.scraper import scrape_site
from type_car_wash.analyzer import classify_car_wash_with_ai

def main():
    print("🚗 Car Wash Scraper & AI Classifier 🚗")
    
    if len(sys.argv) > 1:
        query = sys.argv[1].strip()
    else:
        query = input("Enter the car wash URL or Name: ").strip()
    
    if not query:
        print("Please provide a valid URL or Name.")
        return
        
    url = query
    # If the user didn't provide a URL (no http and no common tld), assume it's a name and search for it
    if not query.startswith(('http://', 'https://')) and '.' not in query:
        from type_car_wash.finder import find_official_website
        
        found_url, confidence = find_official_website(query)
        if not found_url:
            print("Could not confidently find an official website.")
            return
            
        url = found_url
        
    # Ensure URL formatting
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    # Step 1: Scrape
    text = scrape_site(url)
    if not text:
        return
        
    print(f"\n[+] Successfully scraped website. Content length: {len(text)}")
    
    #print(text)
    # Step 2: Classify using AI
    classification_result = classify_car_wash_with_ai(text)
    
    if classification_result:
        print("\n" + "="*50)
        print("                🤖 AI ANALYSIS RESULTS 🤖")
        print("="*50)
        print(f"🏗️  Primary Type:   {classification_result.primary_type}")
        if classification_result.secondary_types:
            print(f"   Secondary Types: {', '.join(classification_result.secondary_types)}")
        print(f"🎯 Confidence:      {classification_result.confidence_score:.0%}")
        print(f"📦 Found Packages:  {', '.join(classification_result.found_packages) if classification_result.found_packages else 'None found'}")
        print(f"🔍 Markers:         {', '.join(classification_result.detected_markers) if classification_result.detected_markers else 'None found'}")
        print(f"🧠 Reasoning:       {classification_result.reasoning}")
        print("="*50 + "\n")
    else:
         print("\n[-] Could not classify the car wash.")

if __name__ == "__main__":
    main()

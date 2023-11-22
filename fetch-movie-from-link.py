# Work in progress

from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import torch
import requests
from bs4 import BeautifulSoup
import sys
import re
import json

# Get the command-line arguments
args = sys.argv

if len(args) != 2:
    print("Usage: python3 " + args[0] + " <filename>", sys.stderr)
    print("Where <filename> is a text file containing a list of links to the movies.", sys.stderr)
    print("Example: python3 " + args[0] + " links.txt", sys.stderr)
    print("The script will output the JSON data of each link in a file with the same name as the "
          + "html file but with.json extension in ./output folder. Movie covers are downloaded to ./pictures", sys.stderr)

    sys.exit(1)

# Get URL list from the file
url_list = []
# with open(args[1]) as f:
with open('links.txt') as f:
    for line in f:
        url_list.append(line.strip())

if len(url_list) == 0:
    print("No URLs found in file", file=sys.stderr)
    sys.exit(1)

# Load the model and tokenizer
# model = "lmsys/vicuna-7b-v1.5-16k"
# model = "lmsys/longchat-7b-v1.5-32k"
model = "meta-llama/Llama-2-13b-chat-hf"
# model = "TheBloke_Llama-2-7B-32K-Instruct-GPTQ"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    device_mapping={
        "cpu": torch.bfloat16,
        "cuda:0": torch.bfloat16,
        "cuda:1": torch.bfloat16,
    }
)

m = AutoModelForCausalLM.from_pretrained(
    model, quantization_config=bnb_config, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model)
text_generator = pipeline(task="text-generation", model=m, tokenizer=tokenizer)

# Process each URL
for url in url_list:
    print("Processing: " + url, file=sys.stderr)
    # Fetch the web page
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the link to the movie cover
    cover_link = soup.select_one("div#movie-main-image-container > a.lightbox").get(
        "href"
    )

    # Extract the image file name from movie image cover url using a regular expression.
    res = re.search("[^/]+$", cover_link)
    picture = res.group(0)

    # Download the cover to pictures/ folder
    r = requests.get(cover_link, stream=True)
    with open("pictures/" + picture, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    # Find country flag
    country_flag_url = soup.select_one("img.nflag").get("src")
    # Extract the image file name from country flag url using a regular expression.
    res = re.search("[^/]+$", country_flag_url)
    country_flag = res.group(0)

    # Initialize the prompt string with the instructions for the model
    prompt = """We analize the film data sheet from the following text scrapped from a website about 
cinema, trim the extra spaces and line feeds and then we will generate a data sheet in the
following JSON format:

[title, year, duration, country, directors (array), writers (array), cast (array), 
music (array), genres (array), summary (sinopsis), rating, picture = """ + picture + ", link = " + url + "]\n\nExtracted text:\n```text\n"

    # Find the text elements in the HTML content
    movie_title = soup.select_one(
        "h1#main-title > span", text=True).get_text().strip()
    movie_info = soup.find(class_="movie-info")
    text_elements = movie_info.find_all(string=True)
    rating = soup.select_one("#movie-rat-avg", text=True)

    # Find the title and rating of the movie
    if len(movie_title) > 0:
        prompt += "Main title: " + movie_title.title() + "\n"
    if (rating is not None):
        rating = rating.get_text().strip()
        prompt += "rating: " + rating.strip() + "\n"
    else:
        prompt += "rating: null\n"

    # Print the text of each non-empty element of the movie information section
    for element in text_elements:
        if element.strip() != "":
            prompt += element.strip() + "\n"

#    prompt += "A partir de esta información reescribe la sinopsis de la película quitanto la palabra FILMAFFINITY y usando un estilo "
#    "más ameno y atractivo, de manera que anime al lector a ver la pelicula y la nueva reseña irá en un campo llamado \"summary\"\n"

    # Finally we add the JSON header to the prompt
    prompt += "```\n\nJSON output:\n```json\n"

    print("Prompt:\n\n" + prompt, file=sys.stderr)
    print("\n\nProcessing prompt...\n", file=sys.stderr)

    # Generate the text
    generated_text = text_generator(
        prompt,
        max_new_tokens=2048,
        use_cache=True,
        temperature=0.8,
        #        top_p=0.8,
        #        top_k=40,
        #        num_beams=2,
        # sudo shutdown 0
        #         early_stopping=True,
        do_sample=True,
    )[
        0
    ]["generated_text"]

    # Print out the generated text
    print("Generated text:\n\n" + generated_text, file=sys.stderr)

    # We strip the prompt from the generated text to return only the JSON output
    generated_text = generated_text.replace(prompt, "")

    # Print out generated text removing the escape characters
    json_end_position = generated_text.find("```")

    # Parse JSON text to get the JSON data
    data = json.loads(str(generated_text[:json_end_position]))

    # Save the JSON data to the output file
    with open("output/" + url.split("/")[-1].split(".")[0] + ".json", "w") as f:
        f.write(json.dumps(data, indent=4, ensure_ascii=True))

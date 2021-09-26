#!/usr/bin/env python3

"""
Generates articles in the style of the CCP's hilarious Global Times website.
"""
import sys, os, re, lzma, statistics, random, math, json
from argparse import ArgumentParser

# For pulling apart their stupid website.
from bs4 import BeautifulSoup

# For extracting HTML content into text.
import html2text

"""
Default for phrase state size.
Higher state size = greater accuracy of generated sentences, with lower creativity.
Lower state size = lower accuracy of generated sentences, with more creativity.
"""
DEFAULT_MARKOV_TITLE_STATE_SIZE = 2
DEFAULT_MARKOV_BODY_STATE_SIZE = 3

"""
Default name for the database to use when generating articles.
"""
DEFAULT_DATABASE_NAME = "default"

"""
Default proxy HTTP port.
"""
DEFAULT_PROXY_PORT = 5000

"""
spaCy NLP pipeline wheel filename
"""
SPACY_NLP_MODEL = "en_core_web_trf"

"""
URL of a scumbag website.
"""
GLOBAL_TIMES_URL = "https://www.globaltimes.cn"

"""
Global regular expressions for various operations.
"""
RE_VALID_HTML_FILE = re.compile(r".*\.s?html?")
RE_PARAGRAPH_SPLIT = re.compile(r"\s*\n\s*\n\s*")
RE_PHRASE_SPLIT = re.compile(r"(?<=[^A-Z].[.?!]) +(?=[A-Z])")
RE_HTML_TO_REMOVE = (
	re.compile(r"<p\s*[^>]*style\s*=.*</p>", re.IGNORECASE),
	re.compile(r"<br>\s*Global Times\s*<br>", re.IGNORECASE),
	re.compile(r"[‹|›]")
)
RE_TEXT_TO_REMOVE = (
	re.compile(r"\\"),
	re.compile(r"^ *\-\s*", re.MULTILINE)
)

"""
Utility class to hide writes to stdout.
"""
class HiddenStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

"""
Class for information relating to a Markov chain node.
"""
class MarkovNode(dict):
	def __init__(self, article_id):
		self.head_count = 0
		self.tail_count = 0
		self.article_id = article_id

	"""
	Increment how many times this Markov node is the head of a phrase.
	"""
	def add_head(self):
		self.head_count += 1

	"""
	Increment how many times this Markov node is the tail of a phrase.
	"""
	def add_tail(self):
		self.tail_count += 1

"""
Simple class for Markov compilation and generation.

Can handle article titles and body sentences. Is also able to correctly determine validity of inserted quotations.
"""
class MarkovChain(object):
	def __init__(self, state_size):
		self.state_size = state_size
		self.words = {}
		self.word_counts = []
		self.word_counts_mean = 0
		self.word_counts_std_dev = 0
		self.paragraph_counts = []
		self.paragraph_counts_mean = 0
		self.paragraph_counts_std_dev = 0
		self.paragraph_phrase_counts = []
		self.paragraph_phrase_counts_mean = 0
		self.paragraph_phrase_counts_std_dev = 0
		self.phrase_word_counts = []
		self.phrase_word_counts_mean = 0
		self.phrase_word_counts_std_dev = 0
		self.shortest_phrase = 0
		self.article_keywords = {}

	"""
	Generate how many paragraphs should be generated based on the compiled content.
	"""
	def generate_num_paragraphs(self):
		return round(max(1, random.gauss(self.paragraph_counts_mean, self.paragraph_counts_std_dev)))

	"""
	Generate how many phrases should exist in a paragraph based on the compiled content.
	"""
	def generate_num_paragraph_phrases(self):
		return round(max(1, random.gauss(self.paragraph_phrase_counts_mean, self.paragraph_phrase_counts_std_dev)))

	"""
	Generate how many words should exist in a phrase based on the compiled content.
	"""
	def generate_num_phrase_words(self):
		return round(max(self.state_size, random.gauss(self.phrase_word_counts_mean, self.phrase_word_counts_std_dev)))

	"""
	Feed words into a simple stack to determine if all quote marks are valid.
	"""
	def validate_quote_stack(self, words):
		stack = 0

		for word in words:
			word = word.strip(".,?-()")

			if word.startswith("\""):
				stack += 1

			elif word.endswith("\""):
				if stack == 0:
					return False

				stack -= 1

		if stack > 0:
			return False

		return True

	"""
	Feed words into a simple stack to determine if all parentheses are valid.
	"""
	def validate_parenthesis_stack(self, words):
		stack = 0

		for word in words:
			word = word.strip(".,?-\"")

			if word.startswith("("):
				stack += 1

			elif word.endswith(")"):
				if stack == 0:
					return False

				stack -= 1

		if stack > 0:
			return False

		return True

	"""
	Add keywords to an article.
	"""
	def add_keywords(self, article_id, keywords):
		self.article_keywords.setdefault(article_id, {})

		for keyword, freq in keywords.items():
			self.article_keywords[article_id][keyword] = self.article_keywords[article_id].get(keyword, 0) + freq

	"""
	Adds either an article title, or an article body to the chain.
	"""
	def add_text(self, article_id, text, keywords = {}):
		word_count = 0

		for paragraph_index, paragraph in enumerate(RE_PARAGRAPH_SPLIT.split(text)):
			for phrase_index, phrase in enumerate(RE_PHRASE_SPLIT.split(paragraph)):
				phrase_words = tuple(phrase.split())

				if not phrase_words:
					continue

				word_count += len(phrase_words)

				if len(phrase_words) < self.shortest_phrase or self.shortest_phrase == 0:
					self.shortest_phrase = len(phrase_words)

				self.phrase_word_counts.append(len(phrase_words))

				for i in range(len(phrase_words) - self.state_size + 1):
					key = phrase_words[i : i + self.state_size]
					
					if i + self.state_size < len(phrase_words):
						value = phrase_words[i + self.state_size]

					else:
						value = None

					self.words.setdefault(key, MarkovNode(article_id))

					if value:
						self.words[key][value] = self.words[key].get(value, 0) + 1

						if i == 0:
							self.words[key].add_head()

					else:
						self.words[key].add_tail()

			self.paragraph_phrase_counts.append(phrase_index + 1)

		self.word_counts.append(word_count)
		self.paragraph_counts.append(paragraph_index + 1)
		self.add_keywords(article_id, keywords)

	"""
	Do some house-keeping after all articles have been fed in.
	"""
	def finish(self):
		self.word_counts_mean = sum(self.word_counts) / len(self.word_counts)
		self.paragraph_counts_mean = sum(self.paragraph_counts) / len(self.paragraph_counts)
		self.paragraph_phrase_counts_mean = sum(self.paragraph_phrase_counts) / len(self.paragraph_phrase_counts)
		self.phrase_word_counts_mean = sum(self.phrase_word_counts) / len(self.phrase_word_counts)
		self.word_counts_std_dev = statistics.pstdev(self.word_counts)
		self.paragraph_counts_std_dev = statistics.pstdev(self.paragraph_counts)
		self.paragraph_phrase_counts_std_dev = statistics.pstdev(self.paragraph_phrase_counts)
		self.phrase_word_counts_std_dev = statistics.pstdev(self.phrase_word_counts)

	"""
	Generate a phrase between min_words and max_words length.
	min_words and max_words are a guide only. The algorithm may need to alter these values if no result is possible.
	"""
	def generate_phrase(self, min_words, max_words):
		min_words = max(min_words, self.shortest_phrase)
		min_words = max(min_words, self.state_size)
		max_words = max(max_words, min_words)
		phrase = []
		backtrack_num = 1
		need_head = True
		attempt_num = 1

		while True:
			if len(phrase) < self.state_size:
				need_head = True
				backtrack_num = 1
				phrase = []

			elif tuple(phrase[-self.state_size :]) not in self.words or not self.words[tuple(phrase[-self.state_size :])] or len(phrase) >= max_words:
				phrase = phrase[: -backtrack_num]
				backtrack_num += 1

				continue

			else:
				need_head = False

			if need_head:
				words, word_probs = [], []

				for key in self.words:
					if self.words[key].head_count:
						words.append(key)
						word_probs.append(self.words[key].head_count)

			else:
				word_dict = self.words[tuple(phrase[-self.state_size :])]
				words, word_probs = zip(*word_dict.items())

			word = random.sample(words, 1, counts = word_probs)[0]

			if type(word) is tuple:
				phrase += word

			else:
				phrase.append(word)

			if len(phrase) >= min_words and len(phrase) <= max_words:
				if self.words[tuple(phrase[-self.state_size :])].tail_count > 0 and self.validate_quote_stack(phrase) and self.validate_parenthesis_stack(phrase):
					break

				# In some circumstances there is no solution with this number of max_words because,
				# for example, we get stuck including a quote which may only be valid with a certain minimum
				# number of possible word combinations. So, increase max_words after many attempts.
				else:
					if attempt_num % 1000 == 0:
						max_words += 1

					attempt_num += 1

			if len(phrase) >= max_words:
				phrase = phrase[: -backtrack_num]
				backtrack_num += 1

		return phrase

"""
Clean up malformed HTML.
"""
def clean_html(html):
	for regex in RE_HTML_TO_REMOVE:
		html = regex.sub("", html)

	return html

"""
Clean up extracted words.
"""
def clean_text(text):
	for regex in RE_TEXT_TO_REMOVE:
		text = regex.sub("", text)

	text = text.replace("“", "\"")

	return text

"""
Match requested keywords against article keywords to determine if a match is found or not.
"""
def match_keywords(keywords, article_keywords):
	if not keywords:
		return True

	for keyword in keywords:
		for article_keyword in article_keywords:
			if re.search(r"\b" + re.escape(keyword.lower()) + r"\b", article_keyword.lower()):
				return True

	return False

"""
Configure and setup NLP.
"""
def setup_nlp(prefer_gpu = False):
	import spacy

	if prefer_gpu:
		spacy.prefer_gpu()

	if SPACY_NLP_MODEL not in spacy.cli.info()["pipelines"]:
		print(f"Downloading spaCy NLP English model: {SPACY_NLP_MODEL}...")

		with HiddenStdout():
			spacy.cli.download(SPACY_NLP_MODEL)

		print("Download complete.")

	nlp = spacy.load(SPACY_NLP_MODEL)
	nlp.get_pipe("transformer").model.attrs["flush_cache_chance"] = 0.1

	return nlp

"""
Read Markov database and return title and body chains.
"""
def read_markov_db(db, title_state_size, body_state_size, keywords = {}):
	title_chain = MarkovChain(title_state_size)
	body_chain = MarkovChain(body_state_size)
	database_dir = os.path.join(os.path.dirname(__file__), "databases")
	num_relevant_articles = 0

	with lzma.open(os.path.join(database_dir, f"{db}.db"), "rt") as handle:
		for article_id, line in enumerate(handle):
			data = json.loads(line.strip())

			if match_keywords(keywords, data["keywords"]):
				num_relevant_articles += 1
				title_chain.add_text(article_id, data["title_text"], keywords = data["keywords"])
				body_chain.add_text(article_id, data["body_text"], keywords = data["keywords"])

	if num_relevant_articles == 0:
		if keywords:
			raise Exception("No articles related to your keywords could be generated.")

		else:
			raise Exception("No articles could be generated.")

	title_chain.finish()
	body_chain.finish()

	return title_chain, body_chain

"""
Compile a Markov chain database.
"""
def compile(args):
	if args.keyword_extraction:
		nlp = setup_nlp(args.prefer_gpu)

	else:
		nlp = None

	num_articles = 0

	converter = html2text.HTML2Text()
	converter.ignore_links = True
	converter.ignore_images = True
	converter.ignore_emphasis = True
	converter.ignore_tables = True
	converter.body_width = 0

	database_dir = os.path.join(os.path.dirname(__file__), "databases")

	os.makedirs(database_dir, exist_ok = True)

	with lzma.open(os.path.join(database_dir, f"{args.db}.db"), "wt", preset = 9 | lzma.PRESET_EXTREME) as db_handle:
		for root, _, files in os.walk(args.dir):
			for path in (os.path.join(root, file) for file in files if RE_VALID_HTML_FILE.search(file)):
				with open(path) as handle:
					soup = BeautifulSoup(handle.read(), "html.parser")
					article = soup.select("div.article")

					if not article:
						continue

					title = soup.select("div.article_title")

					if not title:
						continue

					body = soup.select("div.article_content")

					if not body:
						continue

					title_text = clean_text(converter.handle(clean_html(str(title[0])))).strip()
					body_text = clean_text(converter.handle(clean_html(str(body[0])))).strip()
					keywords = {}

					if not title_text or not body_text:
						continue

					if nlp:
						doc = nlp(title_text)

						for noun in doc.noun_chunks:
							keywords[noun.orth_] = keywords.get(noun.orth_, 0) + 1

						doc = nlp(body_text)

						for noun in doc.noun_chunks:
							keywords[noun.orth_] = keywords.get(noun.orth_, 0) + 1

					print(json.dumps({
						"title_text": title_text,
						"body_text": body_text,
						"keywords": keywords
					}, separators = (",", ":")), file = db_handle)

					num_articles += 1

					print(f"\rCompiled {num_articles} articles...", end = "")

	if num_articles > 0:
		print()

	print("Compilation complete.")

"""
Generate a random article from the title and body chains, returning the generated title and body.
"""
def generate_article(title_chain, body_chain):
	num_title_words = title_chain.generate_num_phrase_words()
	title = " ".join(title_chain.generate_phrase(num_title_words, num_title_words))
	body = []

	for _ in range(body_chain.generate_num_paragraphs()):
		body.append([])

		for _ in range(body_chain.generate_num_paragraph_phrases()):
			num_phrase_words_min = body_chain.generate_num_phrase_words()

			while True:
				num_phrase_words_max = body_chain.generate_num_phrase_words()

				if num_phrase_words_min > num_phrase_words_max:
					num_phrase_words_min, num_phrase_words_max = num_phrase_words_max, num_phrase_words_min

				elif num_phrase_words_min != num_phrase_words_max:
					break

			phrase = " ".join(body_chain.generate_phrase(num_phrase_words_min, num_phrase_words_max))

			# Sometimes their articles are poorly formatted because they stink. So lets just add a period to ensure a complete sentence.
			if not phrase.endswith("."):
				phrase += "."

			body[-1].append(phrase)

	return title, body

"""
Generate a Global Time article from a compiled database.
"""
def generate(args):
	print("Reading database...")

	try:
		title_chain, body_chain = read_markov_db(args.db, args.title_state_size, args.body_state_size, args.keywords)

	except Exception as e:
		print(e, file = sys.stderr)

		return

	print("Reading complete.")

	title, body = generate_article(title_chain, body_chain)

	print("\u262D " * math.ceil(len(title) / 2))
	print(title)
	print("\u262D " * math.ceil(len(title) / 2))
	print()

	for paragraph in body[: -1]:
		print(" ".join(paragraph))
		print()

	print(" ".join(body[-1]))

"""
Setup and run a proxy Global Times website which displays generated articles.
"""
def proxy(args):
	from queue import Queue
	from threading import Thread

	from flask import Flask, request, Response
	import requests

	print("Reading database...")

	try:
		title_chain, body_chain = read_markov_db(args.db, args.title_state_size, args.body_state_size, args.keywords)

	except Exception as e:
		print(e, file = sys.stderr)

		return

	print("Reading complete.")

	app = Flask(__name__)
	article_buffer = Queue(maxsize = 10000)
	server_started = False

	"""
	Return an article title from our queue.
	"""
	def _get_generated_title():
		article = article_buffer.get()

		article_buffer.put_nowait(article)

		return article["title"]

	"""
	Return an article summary from our queue.
	"""
	def _get_generated_summary(num_words, ellipsis = False):
		article = article_buffer.get()
		summary = []

		article_buffer.put_nowait(article)

		for paragraph in article["body"]:
			for phrase in paragraph:
				summary += phrase.split()

				if len(summary) >= num_words:
					summary = " ".join(summary[: num_words])

					if ellipsis:
						summary += " ..."

					return summary

		summary = " ".join(summary)

		if ellipsis:
			summary += " ..."

		return summary

	"""
	Return a number of article body paragraphs from our queue.
	"""
	def _get_generated_paragraphs(num_paragraphs):
		paragraphs = []

		while len(paragraphs) < num_paragraphs:
			article = article_buffer.get()

			article_buffer.put_nowait(article)

			for paragraph in article["body"]:
				paragraphs.append(" ".join(paragraph))

				if len(paragraphs) >= num_paragraphs:
					return paragraphs

	"""
	Replace original HTML with our own.
	"""
	def _replace_html(html):
		soup = BeautifulSoup(html, "html.parser")

		for node in soup.select("a"):
			if node.has_attr("class") and "new_title_" in " ".join(node["class"]):
				node.string = _get_generated_title()

			elif node.parent.has_attr("class") and any(x in node.parent["class"] for x in ("most_article", "new_title_s", "mid_title", "common_title")):
				node.string = _get_generated_title()

			elif node.parent.parent.has_attr("class") and "most_view" in node.parent.parent["class"]:
				node.string = _get_generated_title()

			if node.has_attr("href"):
				node["href"] = node["href"].replace(GLOBAL_TIMES_URL, request.host_url.rstrip("/"))

		for node in soup.select("div.article_title"):
			node.string = _get_generated_title()

		for node in soup.select("p, div.common_desc"):
			words = (node.string or "").strip().split()
			node.string = _get_generated_summary(len(words), True)

		for node in soup.select("div.article_right"):
			for child_node in node.children:
				if child_node.name in ("em", "a"):
					child_node.extract()

			while True:
				found = False

				for child_node in node.children:
					if not child_node.name and child_node.previous_sibling and not child_node.previous_sibling.name:
						child_node.extract()

						found = True

				if not found:
					break

			num_paragraphs = 0

			for child_node in node.children:
				if not child_node.name:
					num_paragraphs += 1

			paragraphs = _get_generated_paragraphs(num_paragraphs)

			for child_node in node.children:
				if not child_node.name:
					child_node.string.replace_with(paragraphs.pop(0))

		return str(soup)

	"""
	Proxy resources from the Global Times
	"""
	def _proxy(*args, **kwargs):
		new_url = f"{GLOBAL_TIMES_URL}{request.full_path}"

		response = requests.request(
			method = request.method,
			url = new_url,
			headers = {key: value for (key, value) in request.headers if key != "Host"},
			data = request.get_data(),
			cookies = request.cookies,
			allow_redirects = False)

		if response.headers.get('content-type') == "text/html":
			body = _replace_html(response.text)

		else:
			body = response.content

		excluded_headers = ["content-encoding", "content-length", "transfer-encoding", "connection"]
		headers = [(name, value) for (name, value) in response.raw.headers.items() if name.lower() not in excluded_headers]

		return Response(body, response.status_code, headers)

	"""
	Serve any URL to the proxy.
	"""
	@app.errorhandler(404)
	def other(error):
		return _proxy()

	"""
	Generate articles over time and place them in our article queue.
	"""
	while True:
		title, body = generate_article(title_chain, body_chain)

		article_buffer.put({
			"title": title,
			"body": body
		})

		if not server_started:
			Thread(target = app.run, kwargs = {"debug": False, "port": args.port}, daemon = True).start()

			server_started = True

def main():
	parser = ArgumentParser(description = __doc__)
	sub_parser = parser.add_subparsers(metavar = "{compile,generate,proxy}", dest = "cmd", required = True)

	compile_parser = sub_parser.add_parser("compile", help = "compile website HTML data from given directory")
	compile_parser.add_argument("dir", help = "website HTML directory to compile")
	compile_parser.add_argument("db", help = "output name of database")
	compile_parser.add_argument("-k", "--keyword-extraction", action = "store_true", help = "use spaCy NLP to identify keywords when compiling database")
	compile_parser.add_argument("-g", "--prefer-gpu", action = "store_true", help = "prefer GPU (if available) when generating keywords")
	compile_parser.set_defaults(func = compile)

	generate_parser = sub_parser.add_parser("generate", help = "generate article")
	generate_parser.add_argument("db", nargs = "?", default = DEFAULT_DATABASE_NAME, help = f"database to use for generation (default: {DEFAULT_DATABASE_NAME})")
	generate_parser.add_argument("-st", "--title_state_size", metavar = "title-state-size", type = int, default = DEFAULT_MARKOV_TITLE_STATE_SIZE, help = f"chain state size for article titles (defualt: {DEFAULT_MARKOV_TITLE_STATE_SIZE})")
	generate_parser.add_argument("-sb", "--body_state_size", metavar = "body-state-size", type = int, default = DEFAULT_MARKOV_BODY_STATE_SIZE, help = f"chain state size for article bodies (defualt: {DEFAULT_MARKOV_BODY_STATE_SIZE})")
	generate_parser.add_argument("-k", "--keywords", nargs = "*", help = "optional list of keywords to generate article about")
	generate_parser.set_defaults(func = generate)

	proxy_parser = sub_parser.add_parser("proxy", help = "start a proxy website serving Global Times articles")
	proxy_parser.add_argument("db", nargs = "?", default = DEFAULT_DATABASE_NAME, help = f"database to use for generation (default: {DEFAULT_DATABASE_NAME})")
	proxy_parser.add_argument("-st", "--title_state_size", metavar = "title-state-size", type = int, default = DEFAULT_MARKOV_TITLE_STATE_SIZE, help = f"chain state size for article titles (defualt: {DEFAULT_MARKOV_TITLE_STATE_SIZE})")
	proxy_parser.add_argument("-sb", "--body_state_size", metavar = "body-state-size", type = int, default = DEFAULT_MARKOV_BODY_STATE_SIZE, help = f"chain state size for article bodies (defualt: {DEFAULT_MARKOV_BODY_STATE_SIZE})")
	proxy_parser.add_argument("-k", "--keywords", nargs = "*", help = "optional list of keywords to generate article about")
	proxy_parser.add_argument("-p", "--port", type = int, default = DEFAULT_PROXY_PORT, help = f"proxy HTTP port (default: {DEFAULT_PROXY_PORT})")
	proxy_parser.set_defaults(func = proxy)

	args = parser.parse_args()

	args.func(args)

if __name__ == "__main__":
	main()

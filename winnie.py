#!/usr/bin/env python3

"""
Generates articles in the style of the CCP's hilarious Global Times website.
"""
import os, re, pickle, gzip, statistics, random, math
from argparse import ArgumentParser

# For pulling apart their stupid website.
from scrapy.selector import Selector

# For extracting HTML content into text.
import html2text

"""
Default for phrase state size.
Higher state size = greater accuracy of generated sentences, with lower creativity.
Lower state size = lower accuracy of generated sentences, with more creativity.
"""
DEFAULT_MARKOV_STATE_SIZE = 3

"""
Default name for the database to use when generating articles.
"""
DEFAULT_DATABASE_NAME = "default"

"""
Global regular expressions for various operations.
"""
RE_VALID_HTML_FILE = re.compile(r".*\.s?html?")
RE_PARAGRAPH_SPLIT = re.compile(r"\s*\n\s*\n\s*")
RE_PHRASE_SPLIT = re.compile(r"(?<=[^A-Z].[.?]) +(?=[A-Z])")
RE_HTML_TO_REMOVE = [
	re.compile(r"<p\s*[^>]*style\s*=.*</p>", re.MULTILINE | re.IGNORECASE),
	re.compile(r"<br>\s*Global Times\s*<br>", re.IGNORECASE),
	re.compile(r"[‹|›]")
]

"""
Simple class for Markov compilation and generation.

Can handle article titles and body sentences. Is also able to correctly determine validity of inserted quotations.
"""
class MarkovChain(object):
	def __init__(self, state_size = DEFAULT_MARKOV_STATE_SIZE):
		self.state_size = state_size
		self.heads = {}
		self.tails = {}
		self.paragraph_heads = {}
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
		return round(max(1, random.gauss(self.phrase_word_counts_mean, self.phrase_word_counts_std_dev)))

	"""
	Feed words into a simple stack to determine if all quote marks are valid.
	"""
	def validate_quote_stack(self, words):
		stack = 0

		for word in words:
			word = word.strip(".,?-")

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
	Adds either an article title, or an article body to the chain.
	"""
	def add_text(self, text):
		word_count = 0

		for paragraph_index, paragraph in enumerate(RE_PARAGRAPH_SPLIT.split(text)):
			for phrase_index, phrase in enumerate(RE_PHRASE_SPLIT.split(paragraph)):
				phrase_words = tuple(phrase.split())

				if not phrase_words:
					continue

				word_count += len(phrase_words)

				self.phrase_word_counts.append(len(phrase_words))
				self.heads[phrase_words[: self.state_size]] = self.heads.get(phrase_words[: self.state_size], 0) + 1
				self.tails[phrase_words[-self.state_size :]] = self.tails.get(phrase_words[-self.state_size :], 0) + 1

				if phrase_index == 0:					
					self.paragraph_heads[phrase_words[: self.state_size]] = self.paragraph_heads.get(phrase_words[: self.state_size], 0) + 1

				for i in range(len(phrase_words) - self.state_size):
					key = phrase_words[i : i + self.state_size]
					value = phrase_words[i + self.state_size]

					self.words.setdefault(key, {})
					self.words[key][value] = self.words[key].get(value, 0) + 1

			self.paragraph_phrase_counts.append(phrase_index + 1)

		self.word_counts.append(word_count)
		self.paragraph_counts.append(paragraph_index + 1)

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
	"""
	def generate_phrase(self, min_words, max_words):
		phrase = []
		backtrack_num = 1

		while True:
			if len(phrase) < self.state_size:
				word_dict = self.heads

			elif tuple(phrase[-self.state_size :]) not in self.words or len(phrase) >= max_words:
				phrase = phrase[: -backtrack_num]
				backtrack_num += 1

				if len(phrase) < self.state_size:
					word_dict = self.heads

			else:
				word_dict = self.words[tuple(phrase[-self.state_size :])]

			words, word_probs = zip(*word_dict.items())
			word = random.sample(words, 1, counts = word_probs)[0]

			if type(word) is tuple:
				phrase += word

			else:
				phrase.append(word)

			if len(phrase) >= min_words and len(phrase) <= max_words:
				if tuple(phrase[-self.state_size :]) in self.tails and self.validate_quote_stack(phrase):
					break

			if len(phrase) >= max_words:
				phrase = phrase[: -backtrack_num]
				backtrack_num += 1

		return phrase

"""
Clean up malformed HTML.
"""
def clean(html):
	for regex in RE_HTML_TO_REMOVE:
		html = regex.sub("", html)

	return html

"""
Compile a Markov chain database.
"""
def compile(args):
	title_chain = MarkovChain(args.state_size)
	body_chain = MarkovChain(args.state_size)

	converter = html2text.HTML2Text()
	converter.ignore_links = True
	converter.ignore_images = True
	converter.ignore_emphasis = True
	converter.ignore_tables = True
	converter.body_width = 0

	for root, _, files in os.walk(args.dir):
		for path in (os.path.join(root, file) for file in files if RE_VALID_HTML_FILE.search(file)):
			with open(path) as handle:
				selector = Selector(text = handle.read(), type = "html")
				article = selector.css("div.article")

				if not article:
					continue

				title = article.css("div.article_title")

				if not title:
					continue

				body = article.css("div.article_content")

				if not body:
					continue

				title_text = converter.handle(title.get()).strip()
				body_text = converter.handle(clean(body.get())).strip()

				if not body_text:
					continue

				title_chain.add_text(title_text)
				body_chain.add_text(body_text)

	title_chain.finish()
	body_chain.finish()

	database_dir = os.path.join(os.path.dirname(__file__), "databases")

	os.makedirs(database_dir, exist_ok = True)

	with gzip.open(os.path.join(database_dir, f"{args.db}.db"), "wb") as handle:
		handle.write(pickle.dumps([title_chain, body_chain]))

"""
Generate a Global Time article from a compiled database.
"""
def generate(args):
	database_dir = os.path.join(os.path.dirname(__file__), "databases")

	with gzip.open(os.path.join(database_dir, f"{args.db}.db"), "rb") as handle:
		title_chain, body_chain = pickle.load(handle)

	num_title_words = title_chain.generate_num_phrase_words()
	title = " ".join(title_chain.generate_phrase(num_title_words, num_title_words))
	paragraphs = []

	for _ in range(body_chain.generate_num_paragraphs()):
		paragraphs.append([])

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

			paragraphs[-1].append(phrase)

	print("\u262D " * math.ceil(len(title) / 2))
	print(title)
	print("\u262D " * math.ceil(len(title) / 2))
	print()
	print()

	for paragraph in paragraphs[: -1]:
		print(" ".join(paragraph))
		print()

	print(" ".join(paragraphs[-1]))

def main():
	parser = ArgumentParser(description = __doc__)
	sub_parser = parser.add_subparsers()
	
	compile_parser = sub_parser.add_parser("compile", help = "compile website HTML data from given directory")
	compile_parser.add_argument("dir", help = "website HTML directory to compile")
	compile_parser.add_argument("db", help = "output name of database")
	compile_parser.add_argument("-s", "--state_size", metavar = "state-size", type = int, default = DEFAULT_MARKOV_STATE_SIZE, help = f"chain state size (defualt: {DEFAULT_MARKOV_STATE_SIZE})")
	compile_parser.set_defaults(func = compile)

	generate_parser = sub_parser.add_parser("generate", help = "generate article")
	generate_parser.add_argument("db", nargs = "?", default = DEFAULT_DATABASE_NAME, help = f"database to use for generation (default: {DEFAULT_DATABASE_NAME})")
	generate_parser.set_defaults(func = generate)

	args = parser.parse_args()

	args.func(args)

if __name__ == "__main__":
	main()

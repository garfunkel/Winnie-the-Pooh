# nonsense
CCP Global Times Article Generator

## Purpose
This is what you get when you plug the bullshit from the CCP's Global Times website into a Markov chain to generate equally dumb (but also somehow dramatically more inteligent) articles.

This came about because I wanted to see how hilarious machine generated articles about Australia would turn out. The wankers at the Global Times write endless reams of immature, offensive literary sewrage about virtually anything the CCP dislikes. So much so that their website appears to be a prime candidate, perfect for this sort of study.

## Usage
### General usage information:
```
usage: winnie.py [-h] {compile,generate} ...

Generates articles in the style of the CCP's hilarious Global Times website.

positional arguments:
  {compile,generate}
    compile           compile website HTML data from given directory
    generate          generate article

optional arguments:
  -h, --help          show this help message and exit
```

### Downloading Content
To generate Global Times articles, first a database must be compiled. An existing database has been created and exists in this repository.

A database is compiled from a directory containing downloaded Global Times HTML articles. You can download a large proportion of their site using a simple `wget` command:
```
wget \
	--recursive \
	--convert-links \
	--domains www.globaltimes.cn \
	--no-parent \
	--accept-regex=".*s?html?" \
	-P MY_DOWNLOAD_DIR
	https://www.globaltimes.cn
```

Let that command run for a few hours. It's best if you hammer their website as hard as you can, obviously. ;)

### Compiling a Database
Once article files have been downloaded, you can compile a database using the following command:
```
python winnie.py compile MY_DOWNLOAD_DIR MY_DATABASE
```

When compiling your database, you may also specify the `state-size`. This value is used to determine the size of Markov chain nodes. Generally speaking, the higher the number the more 'accurate' generated content will be, however creativity will be reduced. A good value for `state-size` is around 2-3. 3 is the default. Setting a high value can also result in slower article generation. You can set your own `state-size` using:
```
python winnie.py compile -s MY_STATE_SIZE MY_DOWNLOAD_DIR MY_DATABASE

```

### Generating Articles
Once a database has been compiled, you can generate articles using a single command. It's important to note that this may take some time - especially if your compiled database uses a higher `state-size` value.
```
python winnie.py generate MY_DATABASE
```

## Possibilities
* It would be cool to produce a proxy website that takes the homepage and/or article pages from the Global Times website, and replaces the content with generated data from this program. It would also be possible to 'tag' phrases and image captions as they are compiled. This would allow images to be selected and used on article pages.
* As sentences are generated independently, they don't track context or meaning. It would be a great idea to tag nouns/verbs/topics (using spaCy or similar) and use that information when generating sentences.

Pull requests are obviously very welcome!

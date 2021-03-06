# Winnie-the-Pooh
CCP Global Times article generator.

## Purpose
This is what you get when you plug the bullshit from the CCP's Global Times website into a Markov chain to generate equally dumb (but also somehow dramatically more inteligent) articles.

This came about because I wanted to see how hilarious machine generated articles about Australia would turn out. The wankers at the Global Times write endless reams of immature, offensive literary sewrage about virtually anything the CCP dislikes. So much so that their website appears to be a prime candidate, perfect for this sort of study.

## Usage
### General usage information:
```
usage: winnie.py [-h] {compile,generate,proxy} ...

Generates articles in the style of the CCP's hilarious Global Times website.

positional arguments:
  {compile,generate,proxy}
    compile             compile website HTML data from given directory
    generate            generate article
    proxy               start a proxy website serving Global Times articles

optional arguments:
  -h, --help            show this help message and exit
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
  -P MY_DOWNLOAD_DIR \
  https://www.globaltimes.cn
```

Let that command run for a few hours. It's best if you hammer their website as hard as you can, obviously. ;)

### Compiling a Database
```
usage: winnie.py compile [-h] [-k] [-g] dir db

positional arguments:
  dir                   website HTML directory to compile
  db                    output name of database

optional arguments:
  -h, --help            show this help message and exit
  -k, --keyword-extraction
                        use spaCy NLP to identify keywords when compiling database
  -g, --prefer-gpu      prefer GPU (if available) when generating keywords
```

Once article files have been downloaded, you can compile a database using the following command:
```
python winnie.py compile MY_DOWNLOAD_DIR MY_DATABASE
```

Winnie-the-Pooh supports keyword extraction via the spaCy NLP library. This allows us to extract keywords/topics from articles, and then use this information to generate more relevant articles. This process can be extremely slow and can take hours when compiling a very large database. You can enmable keyword extraction with the `-k, --keyword-extraction` argument. Keyword extraction can also be performed on some types of GPUs, speeding up the process dramatically. If you would like to use the GPU (if available and supported), you can do so with the `-g, --prefer-gpu` argument.

### Generating Articles
```
usage: winnie.py generate [-h] [-st TITLE_STATE_SIZE] [-sb BODY_STATE_SIZE] [-k [KEYWORDS ...]] [-n NUMBER]
                          [-a [HOST ...]] [-K API_KEY]
                          [db]

positional arguments:
  db                    database to use for generation (default: default)

optional arguments:
  -h, --help            show this help message and exit
  -st TITLE_STATE_SIZE, --title_state_size TITLE_STATE_SIZE
                        chain state size for article titles (defualt: 2)
  -sb BODY_STATE_SIZE, --body_state_size BODY_STATE_SIZE
                        chain state size for article bodies (defualt: 3)
  -k [KEYWORDS ...], --keywords [KEYWORDS ...]
                        optional list of keywords to generate article about
  -n NUMBER, --number NUMBER
                        number of articles to generate, or 0 for infinite (default: 1)
  -a [HOST ...], --api-post [HOST ...]
                        send articles to proxy host(s) via a HTTP POST API call
  -K API_KEY, --api-key API_KEY
                        API key for when sending articles to proxy host(s)
```

Once a database has been compiled, you can generate articles using a single command. It's important to note that this may take some time - especially if your compiled database is very large, as the database will be read into memory first.
```
python winnie.py generate MY_DATABASE
```

When generating articles, you may also specify the `-st, --title-state-size` and `-sb, --body-state-size` arguments. These values are used to determine the size of Markov chain nodes. Generally speaking, the higher the number the more 'accurate' generated content will be, however creativity will be reduced. A good value is around 2-3. 2 is the default for article titles, while 3 is the default for article bodies.

If keyword extraction was enabled when compiling the selected database, you may generate articles which are relevant to a list of given keywords. These keywords can be specified with the `-k, --keywords` argument.

You can generate any number of articles by using the `-n, --number` argument. By specifying 0, an infinite number of articles will be generated.

Finally, you may also send generated articles to other Winnie-the-Pooh instance running in proxy mode by specifying host URLs to the `-a, --api-post` argument. Every time an article has been generated, a call will be made to each of the hosts specified, adding the article for the users of the proxy to enjoy. When using this feature, you may also specify the `-K, --api-key` argument to set a key used for authentication.

### Proxy Website
```
usage: winnie.py proxy [-h] [-st TITLE_STATE_SIZE] [-sb BODY_STATE_SIZE] [-k [KEYWORDS ...]] [-H HOST] [-p PORT]
                       [-c SSL_CERT] [-P SSL_PRIVATE_KEY] [-a] [-K API_KEY]
                       [db]

positional arguments:
  db                    database to use for generation (default: default)

optional arguments:
  -h, --help            show this help message and exit
  -st TITLE_STATE_SIZE, --title_state_size TITLE_STATE_SIZE
                        chain state size for article titles (defualt: 2)
  -sb BODY_STATE_SIZE, --body_state_size BODY_STATE_SIZE
                        chain state size for article bodies (defualt: 3)
  -k [KEYWORDS ...], --keywords [KEYWORDS ...]
                        optional list of keywords to generate article about
  -H HOST, --host HOST  proxy HTTP host (default: localhost)
  -p PORT, --port PORT  proxy HTTP port (default: 5000)
  -c SSL_CERT, --ssl-cert SSL_CERT
                        SSL certificate for HTTPS
  -P SSL_PRIVATE_KEY, --ssl-private-key SSL_PRIVATE_KEY
                        SSL private key for HTTPS
  -a, --api-only        ignore database and do not generate articles - rely on API for articles to be added
  -K API_KEY, --api-key API_KEY
                        API key for when receiving articles from generators
```

Winnie-the-Pooh includes a web server that proxies the Global Times' website, replacing article titles/summaries/bodies in real-time as you view it. This means that you can effectively have your own Global Times in your pocket ready to go, whenever you need a good laugh.

The `-H, --host` argument can be used to set the host IP address. Meanwhile, the `-p, --port` argument can be used to bind the server to a custom network port.

Additionally, you may enable HTTPS support by setting both the `-c, --ssl-cert` and `-P, --ssl-private-key` arguments. When these two are set, all HTTP clients will be forced to connect via HTTPS.

The `-a, --api-only` argument can be used to disable reading of compiled databases and instead rely entirely on other Winnie-the-Pooh instances to send articles through the API.

Finally, the `-K, --api-key` argument may also be used to set the authentication key for the proxy website.

![Behold the mentally deranged glory of our own Global Times](proxy.jpg)

### Generated Article Example
Title:
```
China's Hong Kong Airlines risks having license revoked as body parts found
```

Body:
```
China's so-called "state TV" will divide many Chinese today, experts have said. 

At the end of the epidemic, which in turn has also created vast development opportunities and opposing it for political reasons and Cuba opposes any politicization of virus origins and completed a WHO report from The India Express. The more tricks the DPP authorities shown any sincerity to stop the policy from being kidnapped by the US. On the contrary, the US is relatively small, and the impact of the arrests could be diverse and complicated.

In terms of comprehensive national strength, technology, and many other rights of the trainees and to negate the legitimacy of the Afghan civilian casualties have been reported. Among the athletes, 138 have previous Olympics experience, accounting for 32.02 percent of the cotton and textile industries. Some also worried that Japan is no longer heavily relying on US social media app Musical.ly, which it then merged into TikTok.

In the South China Sea amid her roadshow in Singapore. No matter what the motivation for the US to correct its mistakes, control divergence, and push for higher-level development of Germany-China relations, which is already reporting a profit only five years later, Yu bought a second car, a Tiguan, a Volkswagen SUV.

...
```

## Contributions
Pull requests are obviously very welcome!

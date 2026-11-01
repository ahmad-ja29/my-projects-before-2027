I created tts.py to make audiobooks. I designed it to read out the page numbers so I can easily resume reading.

The code uses 3 processes because, when running it, I realized there was a single-thread bottleneck and my GPU's utilization wasn't high—thanks to Ubuntu's System Monitor, which shows all thread utilization. 

So I implemented 3 processes, but now I realize that I should have used 2 processes with a language like C++, because Python processes consume a lot of RAM.

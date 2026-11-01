import os
import re
import fitz
import soundfile as sf
import multiprocessing as mp
import subprocess
import onnxruntime as ort
from concurrent.futures import ProcessPoolExecutor
from kokoro_onnx import Kokoro

os.environ["ORT_LOGGING_LEVEL"] = "3"

BOOK_DIR = "/home/nnnn/mine/tts/book"
OUTPUT_DIR = "/dev/shm/kokoro_pages"
FINAL_OUTPUT = "/dev/shm/final_output.webm"
ONNX_MODEL_PATH = "/home/nnnn/mine/tts/kokoro-v0_19.onnx"
VOICES_BIN_PATH = "/home/nnnn/mine/tts/voices.bin"

def find_pdf():
    if not os.path.exists(BOOK_DIR):
        raise Exception(f"❌ Directory not found: {BOOK_DIR}")
    pdf_files = [f for f in os.listdir(BOOK_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise Exception(f"❌ No PDF found in {BOOK_DIR}")
    return os.path.join(BOOK_DIR, pdf_files[0])

def safe_chunk_text(text, max_chars=400):
    """Split very long texts at punctuation marks to prevent VRAM memory overflow"""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[.?!;,])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = (current_chunk + " " + sentence).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def parse_single_page(args):
    pdf_path, page_num = args
    with fitz.open(pdf_path) as doc:
        raw_text = doc.load_page(page_num - 1).get_text()
        if not raw_text.strip():
            return page_num, []
        
        # Split at completely empty lines
        raw_blocks = [b.strip() for b in re.split(r'\n\s*\n', raw_text) if b.strip()]
        
        items = []
        # Add page number as an independent paragraph at the beginning
        items.append(f"Page {page_num}.")
        
        for b in raw_blocks:
            clean_block = " ".join(b.split())
            if clean_block:
                # Ensure paragraph length safety to prevent VRAM errors
                safe_chunks = safe_chunk_text(clean_block)
                items.extend(safe_chunks)
                
        return page_num, items

def dual_reader_producer(text_queue, pdf_path, start_page, end_page):
    pages = list(range(start_page, end_page + 1))
    tasks = [(pdf_path, p) for p in pages]
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        for page_num, items in executor.map(parse_single_page, tasks):
            if items:
                text_queue.put((page_num, items))

    text_queue.put(None)

def tts_gpu_worker(text_queue, audio_queue):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    
    session = ort.InferenceSession(
        ONNX_MODEL_PATH, 
        sess_options=opts,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    kokoro = Kokoro.from_session(session, VOICES_BIN_PATH)

    global_idx = 0
    ahead_alert_printed = False

    while True:
        q_size = text_queue.qsize()
        if q_size >= 10 and not ahead_alert_printed:
            print(f"[Buffer] Reader workers are {q_size} pages ahead of GPU.", flush=True)
            ahead_alert_printed = True
        elif q_size < 10:
            ahead_alert_printed = False

        item = text_queue.get()
        if item is None:
            break
        
        page_num, items = item
        page_dir = os.path.join(OUTPUT_DIR, f"page_{page_num}")
        os.makedirs(page_dir, exist_ok=True)

        for p_idx, text in enumerate(items):
            try:
                samples, sample_rate = kokoro.create(
                    text, 
                    voice='af', 
                    speed=1.1, 
                    lang="en-us"
                )
                wav_path = os.path.join(page_dir, f"chunk_{p_idx:04d}.wav")
                sf.write(wav_path, samples, sample_rate)
                audio_queue.put((global_idx, wav_path))
                global_idx += 1
            except Exception as e:
                print(f"⚠️ Skipped chunk due to error: {e}", flush=True)

    audio_queue.put(None)

def encode_single_file(item):
    idx, wav_path = item
    webm_path = wav_path.replace(".wav", ".webm")
    cmd = ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "32k", webm_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(wav_path):
        os.remove(wav_path)
    return (idx, webm_path)

def encoder_manager(audio_queue):
    num_workers = max(1, mp.cpu_count() - 2)
    pool = mp.Pool(processes=num_workers)
    encoded_files = []

    def fetch_items():
        while True:
            item = audio_queue.get()
            if item is None:
                break
            yield item

    for result in pool.imap_unordered(encode_single_file, fetch_items()):
        encoded_files.append(result)
        
    pool.close()
    pool.join()

    encoded_files.sort(key=lambda x: x[0])
    
    list_file = "/dev/shm/concat_list.txt"
    with open(list_file, "w") as f:
        for _, webm_path in encoded_files:
            f.write(f"file '{webm_path}'\n")

    if os.path.exists(list_file) and os.path.getsize(list_file) > 0:
        merge_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", FINAL_OUTPUT]
        subprocess.run(merge_cmd)
        print(f"🎧 Final output file: {FINAL_OUTPUT}")

def main():
    mp.set_start_method('spawn', force=True)

    try:
        pdf_path = find_pdf()
        print(f"📖 Found: {pdf_path}")
    except Exception as e:
        print(e)
        return

    try:
        start_page = int(input("Start page (from 1): "))
        end_page = int(input("End page: "))
    except ValueError:
        print("❌ Invalid input.")
        return

    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    else: 
        os.system(f"rm -rf {OUTPUT_DIR}/*")

    text_queue = mp.Queue(maxsize=512)
    audio_queue = mp.Queue(maxsize=512)
    
    producer_p = mp.Process(target=dual_reader_producer, args=(text_queue, pdf_path, start_page, end_page))
    tts_p = mp.Process(target=tts_gpu_worker, args=(text_queue, audio_queue))
    encoder_p = mp.Process(target=encoder_manager, args=(audio_queue,))
    
    producer_p.start()
    tts_p.start()
    encoder_p.start()
    
    producer_p.join()
    tts_p.join()
    encoder_p.join()
    print("\n✅ Finished successfully.")

if __name__ == "__main__":
    main()

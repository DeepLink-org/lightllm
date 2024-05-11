import psutil
 
for proc in psutil.process_iter(['pid', 'cmdline']):
    if proc.info['cmdline'] and 'python' in proc.info['cmdline'][0]:
        p = psutil.Process(proc.info['pid'])
        p.kill()

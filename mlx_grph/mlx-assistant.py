import time
from string import Template
from mlx_lm import load, generate
from pynput import keyboard
from pynput.keyboard import Key, Controller
import pyperclip

controller = Controller()

MLX_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

PROMPT_TEMPLATE = Template(
    """Fix all typos and casing and punctuation in this text, but preserve all new line characters:

$text

Return only the corrected text, don't include a preamble.
    """
)

print("Starting AI powered typing assistant...")

model, tokenizer = load(MLX_MODEL)

def fix_text(text):
    print(f"Sending text to model: {text}")
    prompt = PROMPT_TEMPLATE.substitute(text=text)
    response = generate(model, tokenizer, prompt=prompt, verbose=True)
    corrected_text = response.strip()
    print(f"Model returned corrected text: {corrected_text}")
    return corrected_text

def fix_current_line():
    print("Triggered fix for current line.")
    # macOS short cut to select current line: Cmd+Shift+Left
    controller.press(Key.cmd)
    controller.press(Key.shift)
    controller.press(Key.left)
    controller.release(Key.cmd)
    controller.release(Key.shift)
    controller.release(Key.left)
    fix_selection()
    
def fix_selection():
    print("Triggered fix for selection.")
    # 1. Copy selection to clipboard
    with controller.pressed(Key.cmd):
        controller.tap("c")
    
    # 2. Get the clipboard string
    time.sleep(0.1)
    text = pyperclip.paste()
    print(f"Text copied to clipboard: {text}")
    
    # 3. Fix string
    if not text:
        print("No text to fix.")
        return
    
    fixed_text = fix_text(text)
    if not fixed_text:
        return
    
    # 4. Paste the fixed string to the clipboard
    pyperclip.copy(fixed_text)
    time.sleep(0.1)
    
    # 5. Paste the clipboard and replace the selected text
    with controller.pressed(Key.cmd):
        controller.tap("v")
    print("Replaced text with corrected version.")
    
def improve_text(text):
    print(f"Sending text to model: {text}")
    prompt = PROMPT_TEMPLATE.improve(text=text)
    response = generate(model, tokenizer, prompt=prompt, verbose=True)
    improved_text = response.strip()
    print(f"Model returned improved text: {improved_text}")
    return improved_text

def improve_current_line():
    print("Triggered fix for current line.")
    # macOS short cut to select current line: Cmd+Shift+Left
    controller.press(Key.cmd)
    controller.press(Key.shift)
    controller.press(Key.left)
    controller.release(Key.cmd)
    controller.release(Key.shift)
    controller.release(Key.left)
    improve_selection()

def improve_selection():
    print("Triggered improvement for selection.")
    # 1. Copy selection to clipboard
    with controller.pressed(Key.cmd):
        controller.tap("c")
    
    # 2. Get the clipboard string
    time.sleep(0.1)
    text = pyperclip.paste()
    print(f"Text copied to clipboard: {text}")
    
    # 3. Fix string
    if not text:
        print("No text to improve.")
        return
    
    improved_text = improve_text(text)
    if not improved_text:
        return
    
    # 4. Paste the fixed string to the clipboard
    pyperclip.copy(improved_text)
    time.sleep(0.1)
    
    # 5. Paste the clipboard and replace the selected text
    with controller.pressed(Key.cmd):
        controller.tap("v")
    print("Replaced text with improved version.")


def on_f5():
    fix_current_line()

def on_f6():
    fix_selection()
    
def on_alt_f5():
    fix_current_line()

def on_alt_f6():
    fix_selection()


print("AI powered typing assistant is running in the background.")
print("Press F5 to fix the current line, F6 to fix the selected text.")
print("Press alt-F5 to improve the current line, alt-F6 to improve the selected text.")
with keyboard.GlobalHotKeys({
    '<f5>': on_f5,
    '<f6>': on_f6,
    '<alt><f5>': on_alt_f5,
    '<alt><f6>': on_alt_f6,
}) as h:
    h.join()

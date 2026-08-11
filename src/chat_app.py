import customtkinter as ctk
import threading
import torch
import os
import sys

# Ensure src is in path if run from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import get_medimicro_300m_model
from inference import stream_safe_response

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class MediMicroChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Medi-Micro Chat")
        self.geometry("900x600")

        # Configure grid layout (1x2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Medi-Micro Settings", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Temperature
        self.temp_label = ctk.CTkLabel(self.sidebar_frame, text="Temperature: 1.0")
        self.temp_label.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.temp_slider = ctk.CTkSlider(self.sidebar_frame, from_=0.0, to=2.0, command=self.update_temp_label)  # type: ignore
        self.temp_slider.set(1.0)
        self.temp_slider.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Top P
        self.topp_label = ctk.CTkLabel(self.sidebar_frame, text="Top-P: 0.9")
        self.topp_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.topp_slider = ctk.CTkSlider(self.sidebar_frame, from_=0.1, to=1.0, command=self.update_topp_label)  # type: ignore
        self.topp_slider.set(0.9)
        self.topp_slider.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Max Tokens
        self.maxt_label = ctk.CTkLabel(self.sidebar_frame, text="Max Tokens: 100")
        self.maxt_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.maxt_slider = ctk.CTkSlider(self.sidebar_frame, from_=10, to=500, number_of_steps=49, command=self.update_maxt_label)
        self.maxt_slider.set(100)
        self.maxt_slider.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="ew")

        # --- Main Chat Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Textbox for chat history
        self.textbox = ctk.CTkTextbox(self.main_frame, state="disabled", wrap="word", font=("Arial", 14))
        self.textbox.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Input Area
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Ask Medi-Micro a medical question...")
        self.entry.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="ew")
        self.entry.bind("<Return>", lambda event: self.send_message())

        self.send_button = ctk.CTkButton(self.input_frame, text="Send", width=80, command=self.send_message)
        self.send_button.grid(row=0, column=1, padx=0, pady=0)

        # Model Loading State
        self.model = None
        self.tokenizer = None
        self.is_generating = False

        self.append_to_chat("System", "Loading Medi-Micro model, please wait...\n")
        self.disable_input()
        
        # Load model in a separate thread so UI doesn't freeze
        threading.Thread(target=self.load_model, daemon=True).start()

    def update_temp_label(self, value):
        self.temp_label.configure(text=f"Temperature: {value:.2f}")

    def update_topp_label(self, value):
        self.topp_label.configure(text=f"Top-P: {value:.2f}")
        
    def update_maxt_label(self, value):
        self.maxt_label.configure(text=f"Max Tokens: {int(value)}")

    def disable_input(self):
        self.entry.configure(state="disabled")
        self.send_button.configure(state="disabled")

    def enable_input(self):
        self.entry.configure(state="normal")
        self.send_button.configure(state="normal")

    def append_to_chat(self, sender, text):
        self.textbox.configure(state="normal")
        self.textbox.insert("end", f"{sender}: " if sender is not None else text)
        self.textbox.see("end")
        self.textbox.configure(state="disabled")

    def load_model(self):
        try:
            from transformers import AutoModelForCausalLM
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            export_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hf_export")
            
            if os.path.exists(export_path):
                from transformers import AutoTokenizer
                self.model = AutoModelForCausalLM.from_pretrained(export_path).to(device)
                self.tokenizer = AutoTokenizer.from_pretrained(export_path)
            else:
                self.after(0, self.append_to_chat, "System", f"Warning: Exported model not found at {export_path}. Run export_bridge.py first.\n")
                return
                
            self.after(0, self.on_model_loaded)
        except Exception as e:
            self.after(0, self.append_to_chat, "System", f"Failed to load model: {str(e)}\n")

    def on_model_loaded(self):
        self.append_to_chat(None, "Model loaded successfully! Ready to chat.\n\n")
        self.enable_input()
        self.entry.focus()

    def send_message(self):
        if self.is_generating:
            return
            
        prompt = self.entry.get().strip()
        if not prompt:
            return

        self.entry.delete(0, "end")
        self.append_to_chat("You", prompt + "\n")
        self.append_to_chat("Medi-Micro", "")
        
        self.is_generating = True
        self.disable_input()

        # Fetch params
        temp = self.temp_slider.get()
        topp = self.topp_slider.get()
        maxt = int(self.maxt_slider.get())

        threading.Thread(target=self.generate_response, args=(prompt, temp, topp, maxt), daemon=True).start()

    def generate_response(self, prompt, temperature, top_p, max_new_tokens):
        try:
            for token in stream_safe_response(self.model, self.tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p):
                self.after(0, self.append_to_chat, None, token)
        except Exception as e:
            self.after(0, self.append_to_chat, None, f"\n[Error]: {str(e)}")
            
        self.after(0, self.on_generation_complete)

    def on_generation_complete(self):
        self.append_to_chat(None, "\n\n")
        self.is_generating = False
        self.enable_input()
        self.entry.focus()

if __name__ == "__main__":
    app = MediMicroChatApp()
    app.mainloop()

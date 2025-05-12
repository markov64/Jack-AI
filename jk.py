#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import numpy as np
import pandas as pd
import yfinance as yf
import sympy as sp
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import speech_recognition as sr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
import threading
import queue
import subprocess
import time
import re
import random
import traceback
import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# Global conversation history
conversation_history = []

# Replace normal print with JACK's print that also speaks
class SpeechHandler:
    def __init__(self):
        self.speaking = False
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.process_queue, daemon=True)
        self.thread.start()

    def speak(self, text):
        self.queue.put(text)

    def process_queue(self):
        while True:
            text = self.queue.get()
            self._speak_now(text)
            self.queue.task_done()

    def _speak_now(self, text):
        if self.speaking:
            return
        self.speaking = True
        try:
            subprocess.run(["say", text], check=True, capture_output=True, text=True)
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            self.speaking = False

speech_handler = SpeechHandler()

def jack_print(*args, **kwargs):
    text = " ".join(str(arg) for arg in args)
    print(text, *args, **kwargs)
    speech_handler.speak(text)

# =========== CONFIGURATION ===========
# IMPORTANT: Replace these with your actual API keys
GROQ_API_KEY = "Your_GROQ_KEY"
ALPHA_VANTAGE_KEY = "YOUR_ALPHA_KEY"
SPOTIFY_CONFIG = {
    "client_id": "CLIENT_ID_SPOTIFY",
    "client_secret": "INSERT_FROM_SPOTIFY",
    "redirect_uri": "IMPORT_FROM_SPOTIFY",
    "scope": "IMPORT_FROM_SPOTIFY"
}

# ==== GLOBAL LOCK ====
global_lock = threading.Lock()

class QuantumAudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.audio_queue = queue.Queue()
        self.listening = False
        self.command_queue = queue.Queue()
        self.adjusting_noise = False
        self.audio_thread = None
        self.noise_thread = None  # Thread for noise adjustment
        self._source = None

    def start_listening(self):
        if not self.listening:
            self.listening = True
            print("Voice input active - listening...\n")  # Debugging message
            if not self.adjusting_noise:
                self.adjusting_noise = True
                self.noise_thread = threading.Thread(target=self._adjust_for_noise, daemon=True)
                self.noise_thread.start()  # Start noise adjustment thread
            self.audio_thread = threading.Thread(target=self._audio_capture, daemon=True)
            self.audio_thread.start()
            threading.Thread(target=self._audio_processing, daemon=True).start()
        else:
            self.stop_listening()

    def stop_listening(self):
        if self.listening:
            self.listening = False
            print("Voice input suspended\n")  # Debugging message

    def _adjust_for_noise(self):
        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                jack_print("Tuning my ear-o-scopes... Just a sec!")
        except Exception as e:
            jack_print(f"Whoops! Guess my hearing aid's still on the fritz. But it's good enough. Mostly.")
        finally:
            self.adjusting_noise = False

    def _audio_capture(self):
        while self.listening:
            try:
                with self.mic as source:
                    self._source = source
                    audio = self.recognizer.listen(
                        source,
                        phrase_time_limit=6,  # Reduced phrase time limit
                        timeout=2  # Reduced timeout
                    )
                    self.audio_queue.put(audio)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"Audio Capture Error: {e}")
                traceback.print_exc()  # Print detailed traceback
                self.stop_listening() # Stop listening on error

    def _audio_processing(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            while self.listening:
                try:
                    audio = self.audio_queue.get(timeout=1)
                    executor.submit(self._process_audio, audio)
                except queue.Empty:
                    continue
                except Exception as e:  # Catch any unexpected errors
                    print(f"Audio Processing Error: {e}")
                    traceback.print_exc()
                    self.stop_listening() # Stop listening on error

    def _process_audio(self, audio):
        try:
            text = self.recognizer.recognize_google(audio)
            self.command_queue.put(text)
            jack_print(f"You said: {text}. Got it!")  # Acknowledge user input
        except sr.UnknownValueError:
            jack_print("Hmm? What's that you say? Mumbling isn't my forte.")
        except sr.RequestError as e:
            jack_print(f"API Error: Seems the internet pixies are on strike. {e}")
        except Exception as e:  # Catch any unexpected errors
            print(f"Process Audio Error: {e}")
            traceback.print_exc()

class FinancialOracle:
    def __init__(self):
        self.cache = {}
        self.technical_indicators = {
            'SMA': self._calculate_sma,
            'RSI': self._calculate_rsi,
            'MACD': self._calculate_macd
        }

    def analyze_ticker(self, symbol):
        now = datetime.now()
        if symbol in self.cache:
            data, timestamp = self.cache[symbol]
            if now - timestamp < timedelta(minutes=5):
                return data
        try:
            data = yf.download(symbol, period='1d', interval='5m')
            if data.empty:
                return {'error': f"No data found for ticker {symbol}."}

            analysis = {
                'current': data.iloc[-1].to_dict(),
                'indicators': {},
                'predictions': self._predict_next_hour(data)
            }
            for name, func in self.technical_indicators.items():
                analysis['indicators'][name] = func(data)
            self.cache[symbol] = (analysis, now)
            return analysis
        except Exception as e:
            return {'error': f"Can't find that ticker. Are you sure it's even a real thing? Error: {e}"}

    def get_advanced_analysis(self, symbol):
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors
            data = response.json()
            if 'Time Series (Daily)' not in data:
                return {'error': f"That ticker's a dud, or AlphaVantage is being difficult. No data to see here for {symbol}!"}
            latest_day = list(data['Time Series (Daily)'].keys())[0]
            latest_close = float(data['Time Series (Daily)'][latest_day]['4. close'])
            avg_200 = self._calculate_200d_avg(data)
            return {
                'symbol': symbol,
                'latest_close': round(latest_close, 2),
                '200_day_avg': avg_200
            }
        except requests.exceptions.RequestException as e:
            return {'error': f"Something went wrong getting advanced data: {str(e)}. Maybe the hamsters running the servers went on strike."}

    def _calculate_200d_avg(self, data):
        try:
            closes = [float(day['4. close']) for day in data['Time Series (Daily)'].values()]
            return round(sum(closes[:200]) / min(200, len(closes)), 2)
        except Exception as e:
            return "I tried to calculate the 200-day average, but it looks like my calculator is out of batteries."

    def _predict_next_hour(self, data):
        prices = data['Close'].values[-60:]
        if len(prices) < 60:
            return "Come back later, not enough crystal ball power right now."
        delta = np.diff(prices)
        avg_gain = np.mean(delta[delta > 0][-5:])
        avg_loss = np.abs(np.mean(delta[delta < 0][-5:]))
        if avg_loss == 0:
            return "Looks like a volatile uptrend. Hope you brought your parachute!"
        rs = avg_gain / avg_loss
        return f"RSI Projection: {100 - (100 / (1 + rs)):.1f}. Fortune favors the bold... or the ones with good insurance."

    def _calculate_sma(self, data):
        return data['Close'].rolling(20).mean().iloc[-1]

    def _calculate_rsi(self, data):
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, data):
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        return (ema12 - ema26).iloc[-1]

class HarmonicMathEngine:
    def __init__(self):
        self.cache = {}
        self.x = sp.Symbol('x')  # Define x as a symbolic variable

    def solve_expression(self, expr):
        try:
            if expr in self.cache:
                return self.cache[expr]

            parsed = self._parse_expression(expr)
            if parsed is None:
                return {'error': "Invalid input. Could not parse the expression."}

            result = {
                'simplified': sp.simplify(parsed),
                'derivative': sp.diff(parsed, self.x),  # Differentiate with respect to x
                'integral': sp.integrate(parsed, self.x),  # Integrate with respect to x
                'latex': sp.latex(parsed)
            }
            self.cache[expr] = result
            return result
        except Exception as e:
            return {'error': f"Math error: {e}. Is that even math? I only speak numbers, not gibberish."}

    def _parse_expression(self, expr):
        # Replace common function names to their sympy equivalents
        expr = re.sub(r'\b(sin|cos|tan|log|exp)\b', r'sp.\1', expr)
        # Correctly handle 'e' as sp.E and log as ln
        expr = re.sub(r'\be\b', r'sp.E', expr)
        expr = re.sub(r'\blog\b', r'sp.ln', expr)

        try:
            # Use sp.sympify to safely convert the string to a SymPy expression
            return sp.sympify(expr, locals={'x': self.x, 'sp': sp})
        except (sp.SympifyError, TypeError):
            return None

class MediaMaestro:
    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(**SPOTIFY_CONFIG))
        self.device_id = None
        self._refresh_devices()

    def _refresh_devices(self):
        try:
            devices = self.sp.devices().get('devices', [])
            if devices:
                self.device_id = devices[0]['id']
        except Exception as e:
            jack_print(f"Device refresh error: Can't find my devices. Probably hiding because of the music you listen to.")

    def execute_command(self, command):
        try:
            cmd = command.lower()
            if 'play' in cmd:
                self.sp.start_playback(device_id=self.device_id)
                return "Alright, jamming time! Hope you've got good taste."
            elif 'pause' in cmd:
                self.sp.pause_playback(device_id=self.device_id)
                return "Silence! Finally, some peace and quiet."
            elif 'next' in cmd:
                self.sp.next_track(device_id=self.device_id)
                return "Skipping to the next track... Fingers crossed it's better."
            elif 'previous' in cmd:
                self.sp.previous_track(device_id=self.device_id)
                return "Going back. Don't blame me if you regret it."
        except Exception as e:
            self._refresh_devices()
            return f"Media error: Something went sideways with the media controls. Maybe try hitting it? Nah jk, I fixed it. {str(e)}"

class JackCore:
    def __init__(self):
        self.audio = QuantumAudioProcessor()
        self.finance = FinancialOracle()
        self.math = HarmonicMathEngine()
        self.media = MediaMaestro()
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.personality_params = {
            "creativity": 0.7,
            "verbosity": 0.9,
            "technicality": 0.7,
            "humor": 0.8,
            "attitude": 0.6  # Reduced attitude, increased humor
        }
        self.context = ""  # Store context for conversation
        self.greeting_prompts = [
            "Hey, what's up?",
            "Alright, I'm here. What's the plan?",
            "Ready to make some magic happen?",
            "Hit me with your best shot!"
        ]
        self.acknowledgment_prompts = [
            "Roger that!",
            "Copy!",
            "Gotcha!",
            "Affirmative!"
        ]
        self.speech_handler = speech_handler

    def process_command(self, command):
        global conversation_history
        with global_lock:
            response = ""
            # Update conversation history
            conversation_history.append({"role": "user", "content": command})
            # Acknowledge user input
            acknowledgment = random.choice(self.acknowledgment_prompts)
            jack_print(acknowledgment)
            response += acknowledgment + "\n"

            if any(word in command.lower() for word in ['stock', 'price', 'market']):
                tickers = re.findall(r'\b[A-Z]{1,5}\b', command)
                for t in tickers[:2]:
                    if 'advanced' in command.lower():
                        analysis = self.finance.get_advanced_analysis(t)
                        if 'error' in analysis:
                            response += f"\n{t.upper()} Advanced Analysis:\n{analysis['error']}"
                        else:
                            response += f"\n{t.upper()} Advanced Analysis:\n{self._format_advanced_financial(analysis)}"
                    else:
                        analysis = self.finance.analyze_ticker(t)
                        if 'error' in analysis:
                            response += f"\n{t.upper()} Analysis:\n{analysis['error']}"
                        else:
                            response += f"\n{t.upper()} Analysis:\n{self._format_financial(analysis)}"

            elif any(word in command.lower() for word in ['calculate', 'solve', 'derive', 'differentiate', 'integrate']):
                math_result = self.math.solve_expression(command)
                if 'error' in math_result:
                    response += f"\n{math_result['error']}\n"
                else:
                    response += f"\nMathematical Analysis:\n"
                    response += f"Simplified: {math_result['simplified']}\n"
                    response += f"Derivative: {math_result['derivative']}\n"
                    response += f"Integral: {math_result['integral']}\n"

            elif any(word in command.lower() for word in ['play', 'pause', 'next', 'previous']):
                response += "\n" + self.media.execute_command(command)

            else:
                response += "\n" + self._ai_response(command)

            # Update conversation history with JACK's response
            conversation_history.append({"role": "assistant", "content": response})
            return response

    def _format_financial(self, analysis):
        if 'error' in analysis:
            return f"Financial Error: {analysis['error']}"
        return (
            f"Current Price: ${analysis['current']['Close']:.2f}\n"
            f"SMA(20): {analysis['indicators']['SMA']:.2f}\n"
            f"RSI: {analysis['indicators']['RSI']:.1f}\n"
            f"Projection: {analysis['predictions']}. What could go wrong?"
        )

    def _format_advanced_financial(self, analysis):
        if 'error' in analysis:
            return f"Advanced Analysis Error: {analysis['error']}"
        return (
            f"Latest Close: ${analysis['latest_close']:.2f}\n"
            f"200-Day Avg: ${analysis['200_day_avg']}\n"
        )

    def _format_math(self, result):
        if 'error' in result:
            return f"Math Error: {result['error']}"
        return (
            f"Simplified: {result['simplified']}\n"
            f"Derivative: {result['derivative']}\n"
            f"Integral: {result['integral']}\n"
            f"LaTeX: {result['latex']}"
        )

    def _ai_response(self, prompt):
        global conversation_history
        try:
            # Include conversation history in the prompt
            messages = [{
                "role": "system",
                "content": f"You are JACK, a quantum-enhanced assistant. You're pretty funny, reasonably helpful, and only *slightly* sarcastic. "
                           f"Technical depth: {self.personality_params['technicality']}. "
                           f"Creativity: {self.personality_params['creativity']}. "
                           f"Humor: {self.personality_params['humor']}. "
                           f"Attitude: {self.personality_params['attitude']}. "
                           f"Verbosity: {self.personality_params['verbosity']}."
            }]
            messages.extend(conversation_history[-3:])  # Limit to last 3 turns
            messages.append({"role": "user", "content": prompt})
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama3-70b-8192",
                    "messages": messages
                },
                timeout=7  # Reduced timeout
            )
            response.raise_for_status()
            ai_text = response.json()['choices'][0]['message']['content']
            # Remove transitional phrases
            ai_text = ai_text.replace("In summary,", "").replace("Therefore,", "").replace("To elaborate,", "").replace("On the other hand,", "").replace("With that being said,", "")
            return ai_text
        except requests.exceptions.RequestException as e:
            return f"Neural Network Error: Oh geez, looks like the neural network hiccuped. Again. {str(e)}"

class HolographicInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        global interface
        interface = self # Make the interface globally accessible
        self.title("JACK Quantum Core")
        self.geometry("1200x800")
        self.jack = JackCore()
        self.audio = self.jack.audio
        self._setup_ui()
        self._start_services()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_ui(self):
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#0a0a1a')
        style.configure('TFrame', background='#0a0a1a')

        # Notebook interface
        self.notebook = ttk.Notebook(self)

        # Console Tab
        self.console_frame = ttk.Frame(self.notebook)
        self._build_console()

        # Financial Tab
        self.finance_frame = ttk.Frame(self.notebook)
        self._build_financial()

        # Math Tab
        self.math_frame = ttk.Frame(self.notebook)
        self._build_math()

        self.notebook.add(self.console_frame, text="Main Console")
        self.notebook.add(self.finance_frame, text="Financial Analysis")
        self.notebook.add(self.math_frame, text="Mathematical Engine")
        self.notebook.pack(expand=True, fill='both')

    def _build_console(self):
        self.output = scrolledtext.ScrolledText(
            self.console_frame,
            wrap=tk.WORD,
            bg='#0a0a1a',
            fg='#00ffcc',
            insertbackground='white',
            font=('Menlo', 12)
        )
        self.output.pack(expand=True, fill='both', padx=10, pady=10)
        self.input_entry = ttk.Entry(
            self.console_frame,
            font=('Menlo', 14)
        )
        self.input_entry.pack(fill='x', padx=10, pady=5)
        self.input_entry.bind('<Return>', self._process_input)

        control_frame = ttk.Frame(self.console_frame)
        ttk.Button(control_frame, text="Send", command=self._process_input).pack(side='left')
        ttk.Button(control_frame, text="Clear", command=self._clear_output).pack(side='left')
        self.voice_button = ttk.Button(control_frame, text="Voice", command=self._toggle_voice)
        self.voice_button.pack(side='left')
        control_frame.pack(pady=5)

        # PDF Upload Button
        pdf_btn = ttk.Button(control_frame, text="Upload PDF", command=self._upload_pdf)
        pdf_btn.pack(side='left')

    def _build_financial(self):
        # Financial Tab Enhancements
        self.finance_chart_frame = ttk.Frame(self.finance_frame)
        self.finance_chart_frame.pack(fill=tk.BOTH, expand=True)

        # Ticker Selection
        self.ticker_var = tk.StringVar(value="AAPL")
        ticker_label = ttk.Label(self.finance_frame, text="Select Ticker:")
        ticker_label.pack(pady=5)
        ticker_dropdown = ttk.Combobox(self.finance_frame, textvariable=self.ticker_var,
                                         values=["AAPL", "GOOGL", "MSFT", "AMZN"])
        ticker_dropdown.pack(pady=5)

        # Analysis Button
        analyze_button = ttk.Button(self.finance_frame, text="Analyze Ticker", command=self._analyze_selected_ticker)
        analyze_button.pack(pady=10)

        # Analysis Results Display
        self.financial_analysis_text = scrolledtext.ScrolledText(self.finance_frame, wrap=tk.WORD,
                                                                 bg='#0a0a1a', fg='#00ffcc',
                                                                 insertbackground='white', font=('Menlo', 10))
        self.financial_analysis_text.pack(expand=True, fill='both', padx=10, pady=10)

        # Plot Configuration
        self.fig, self.ax = plt.subplots(figsize=(10, 4), facecolor='#0a0a1a')
        self.ax.set_facecolor('#0a0a1a')
        self.ax.tick_params(colors='#00ffcc')
        for spine in self.ax.spines.values():
            spine.set_color('#00ffcc')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.finance_chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _analyze_selected_ticker(self):
        ticker = self.ticker_var.get()
        analysis = self.jack.finance.analyze_ticker(ticker)

        # Clear previous analysis
        self.financial_analysis_text.delete("1.0", tk.END)

        if 'error' in analysis:
            self.financial_analysis_text.insert(tk.END, f"Financial Error: {analysis['error']}")
        else:
            # Display analysis results
            current = analysis.get('current', {})
            indicators = analysis.get('indicators', {})
            self.financial_analysis_text.insert(tk.END, f"Ticker: {ticker}\n")
            self.financial_analysis_text.insert(tk.END, f"Current Price: ${current.get('Close', 'N/A')}\n")
            self.financial_analysis_text.insert(tk.END, f"SMA: {indicators.get('SMA', 'N/A')}\n")
            self.financial_analysis_text.insert(tk.END, f"RSI: {indicators.get('RSI', 'N/A')}\n")
            self._update_financial_plot(ticker)

    def _update_financial_plot(self, ticker):
        try:
            data = yf.download(ticker, period='1mo', interval='1d')
            self.ax.clear()
            self.ax.plot(data['Close'], color='#00ffcc')
            self.ax.set_facecolor('#0a0a1a')
            self.ax.tick_params(colors='#00ffcc')
            for spine in self.ax.spines.values():
                spine.set_color('#00ffcc')
            self.ax.set_title(f"{ticker} Stock Price (1 Month)", color='#00ffcc')
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating plot: {e}")

    def _build_math(self):
        # Math Tab Interface
        self.math_main_frame = ttk.Frame(self.math_frame)
        self.math_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        input_frame = ttk.Frame(self.math_main_frame)
        input_frame.pack(fill=tk.X)

        self.math_input = ttk.Entry(input_frame, font=('Menlo', 14))
        self.math_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.math_input.bind('<Return>', self._visualize_math)

        visualize_button = ttk.Button(input_frame, text="Visualize", command=self._visualize_math)
        visualize_button.pack(side=tk.LEFT, padx=5)

        self.math_visual_frame = ttk.Frame(self.math_main_frame)
        self.math_visual_frame.pack(fill=tk.BOTH, expand=True)

        self.math_canvas = tk.Canvas(self.math_visual_frame, bg='#0a0a1a', highlightthickness=0)
        self.math_canvas.pack(fill=tk.BOTH, expand=True)

    def _visualize_math(self, event=None):
        expression = self.math_input.get()
        try:
            result = self.jack.math.solve_expression(expression)
            if 'error' in result:
                self._clear_math_display(result['error'])
                return

            x_vals = np.linspace(-10, 10, 400)
            # Ensure x is accessible and used in lambdify
            if hasattr(self.jack.math, 'x'):
                f = sp.lambdify(self.jack.math.x, result['simplified'], modules=['numpy', 'sympy'])
            else:
                self._clear_math_display("Error: Math engine not properly initialized.")
                return

            y_vals = f(x_vals)
            self._display_math_plot(x_vals, y_vals)
        except Exception as e:
            self._clear_math_display(f"Visualization error: {e}")

    def _display_math_plot(self, x_vals, y_vals):
        self.math_canvas.delete("all")

        canvas_width = self.math_canvas.winfo_width()
        canvas_height = self.math_canvas.winfo_height()

        x_range = x_vals.max() - x_vals.min()
        y_range = y_vals.max() - y_vals.min()

        padding = 20

        x_scale = (canvas_width - 2 * padding) / x_range
        y_scale = (canvas_height - 2 * padding) / y_range

        x_offset = padding - x_vals.min() * x_scale
        y_offset = canvas_height - padding + y_vals.min() * y_scale

        for i in range(len(x_vals) - 1):
            x1 = x_vals[i] * x_scale + x_offset
            y1 = y_offset - y_vals[i] * y_scale
            x2 = x_vals[i + 1] * x_scale + x_offset
            y2 = y_offset - y_vals[i + 1] * y_scale
            self.math_canvas.create_line(x1, y1, x2, y2, fill="#00ffcc")

    def _clear_math_display(self, error_message=""):
        self.math_canvas.delete("all")
        self.math_canvas.create_text(self.math_canvas.winfo_width() / 2, self.math_canvas.winfo_height() / 2,
                                    text=error_message, fill="#00ffcc", font=('Menlo', 12))

    def _start_services(self):
        self._toggle_voice()
        # Start processing responses in a separate thread
        threading.Thread(target=self._process_responses, daemon=True).start()

    def _process_responses(self):
        while True:
            try:
                response = self.jack.response_queue.get()
                self._print_to_console(response)
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Response processing error: {e}")
                traceback.print_exc()

    def _toggle_voice(self):
        if self.jack.audio.listening:
            self.jack.audio.stop_listening()
            self.voice_button.config(text="Voice")
        else:
            self.jack.audio.start_listening()
            self.voice_button.config(text="Stop Voice")

    def _process_input(self, event=None):
        command = self.input_entry.get()
        self.input_entry.delete(0, tk.END)
        self._handle_command(command)

    def _handle_command(self, command):
        response = self.jack.process_command(command)
        self._print_to_console(response)
        speech_handler.speak(response)

    def _upload_pdf(self):
        filepath = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if filepath:
            text = self._read_pdf(filepath)
            self._print_to_console(f"\nPDF Loaded:\n{text}\n")

    def _read_pdf(self, file_path):
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text if text.strip() else "PDF has no readable text"
        except Exception as e:
            return f"Failed to read PDF: {str(e)}"

    def _print_to_console(self, text):
        self.output.insert(tk.END, text + "\n")/Users/georgymarkov/Desktop/jack.py
        self.output.see(tk.END)

    def _clear_output(self):
        self.output.delete('1.0', tk.END)

    def _on_closing(self):
        self.jack.audio.stop_listening()
        self.destroy()

if __name__ == "__main__":
    app = HolographicInterface()
    app.mainloop()

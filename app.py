from flask import Flask, render_template, request, redirect, url_for
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# === Chatbot Setup ===
model_path = "./models/gpt2-chemistry"
tokenizer_chat = GPT2Tokenizer.from_pretrained(model_path)
model_chat = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer_chat.pad_token = tokenizer_chat.eos_token
model_chat.config.pad_token_id = model_chat.config.eos_token_id

# === Quiz Setup ===
quiz_model_path = "./models/best_gpt2_mcq (1).pt"
quiz_dataset_path = "./data/train.csv"
quiz_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
special_tokens = {'pad_token': '<|pad|>', 'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>'}
quiz_tokenizer.add_special_tokens(special_tokens)
quiz_model = GPT2LMHeadModel.from_pretrained("gpt2")
quiz_model.resize_token_embeddings(len(quiz_tokenizer))
quiz_model.load_state_dict(torch.load(quiz_model_path, map_location=torch.device('cpu')))
quiz_model.eval()
quiz_data = pd.read_csv(quiz_dataset_path)

# === Schedule Setup ===
schedule_data = pd.read_csv("./data/train.csv")
topic_difficulty = schedule_data['topic;'].value_counts().reset_index()
topic_difficulty.columns = ['topic', 'count']
total_difficulty = topic_difficulty['count'].sum()

# Helper functions for schedule
def convert_hours_to_hm(hours):
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h}h {m}m" if h > 0 else f"{m}m"

def generate_schedule(hours_per_day, num_days, start_date, end_date):
    total_hours = hours_per_day * num_days
    topic_difficulty['allocated_hours'] = topic_difficulty['count'] / total_difficulty * total_hours

    total_range_days = (end_date - start_date).days + 1
    all_dates = [start_date + timedelta(days=i) for i in range(total_range_days)]
    study_dates = sorted(random.sample(all_dates, num_days))

    schedule = []
    topics_list = topic_difficulty.to_dict('records')
    current_index = 0

    for study_date in study_dates:
        daily_hours = hours_per_day
        day_schedule = []

        while daily_hours > 0 and current_index < len(topics_list):
            topic = topics_list[current_index]
            allocate = min(topic['allocated_hours'], daily_hours)

            if allocate > 0:
                day_schedule.append({
                    "topic": topic['topic'],
                    "duration": convert_hours_to_hm(allocate)
                })
                topics_list[current_index]['allocated_hours'] -= allocate
                daily_hours -= allocate

            if topics_list[current_index]['allocated_hours'] <= 0:
                current_index += 1

        schedule.append({"date": study_date.strftime("%Y-%m-%d"), "study_plan": day_schedule})

    return schedule

# ===== Routes =====

@app.route('/')
def index():
    return render_template('index.html')  

# --- Chatbot route ---
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    answer = None
    if request.method == 'POST':
        question = request.form.get('question', '')
        if question:
            prompt = f"Q: {question}\nA:"
            input_ids = tokenizer_chat.encode(prompt, return_tensors="pt")

            output_ids = model_chat.generate(
                input_ids,
                max_length=700,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer_chat.eos_token_id,
                eos_token_id=tokenizer_chat.eos_token_id,
            )

            decoded = tokenizer_chat.decode(output_ids[0], skip_special_tokens=True)
            try:
                answer = decoded.split("A:")[1].split("Q:")[0].strip()
            except:
                answer = "I'm not sure how to answer that."
    return render_template('chatbot.html', answer=answer)

# --- Quiz routes ---
@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
   
    if request.method == 'POST' and 'num' in request.form:
        num = int(request.form['num'])
        questions = []

        for i in range(num):
            row = quiz_data.sample(1).iloc[0]
            question = row['message_1']
            correct_answer = row['message_2']
            sub_topic = row['sub_topic']

            distractor_pool = quiz_data[quiz_data['message_2'] != correct_answer]['message_2'].tolist()
            distractors = random.sample(distractor_pool, 3)

            options = distractors[:]
            insert_idx = random.randint(0, 3)
            options.insert(insert_idx, correct_answer)

            questions.append({
                'id': i,
                'question': question,
                'options': options,
                'correct_idx': insert_idx,
                'sub_topic': sub_topic
            })

        return render_template('quiz.html', questions=questions)

   
    elif request.method == 'POST':
        answers = request.form.to_dict()
        wrong_topics = []
        results = []

        for key, user_ans in answers.items():
            if not key.startswith("ans_"):
                continue
            qid = int(key.split('_')[1])
            correct_idx = int(request.form[f'correct_{qid}'])
            sub_topic = request.form[f'topic_{qid}']
            correct_letter = chr(65 + correct_idx)

            is_correct = user_ans == correct_letter
            if not is_correct:
                wrong_topics.append(sub_topic)

            results.append({
                'question': request.form[f'question_{qid}'],
                'options': [request.form[f'option_{qid}_{i}'] for i in range(4)],
                'user_ans': user_ans,
                'correct_ans': correct_letter,
                'sub_topic': sub_topic,
                'is_correct': is_correct
            })

        return render_template('quiz_results.html', results=results, wrong_topics=sorted(set(wrong_topics)))

    
    return render_template('quiz_start.html')

# --- Schedule routes ---
@app.route('/schedule', methods=['GET', 'POST'])
def schedule():
    schedule = None
    error = None
    if request.method == 'POST':
        try:
            hours_per_day = float(request.form['hours_per_day'])
            num_days = int(request.form['num_days'])
            start_date = datetime.strptime(request.form['start_date'], "%Y-%m-%d")
            end_date = datetime.strptime(request.form['end_date'], "%Y-%m-%d")

            date_range_days = (end_date - start_date).days + 1
            if num_days > date_range_days:
                error = f"Number of study days ({num_days}) cannot exceed date range ({date_range_days} days)."
            else:
                schedule = generate_schedule(hours_per_day, num_days, start_date, end_date)
        except Exception as e:
            error = str(e)

    return render_template('schedule.html', schedule=schedule, error=error)


if __name__ == '__main__':
    app.run(debug=True)

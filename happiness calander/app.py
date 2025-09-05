import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import tempfile
from io import BytesIO
import time
import datetime
import calendar
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import base64
import os

# Import the model from your new model.py file
from model import EmotionCNN

# Emotion class labels
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTION_EMOJIS = {'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊',
                  'neutral': '😐', 'sad': '😢', 'surprise': '😲'}
EMOTION_COLORS = {
    'angry': '#FF6B6B', 'disgust': '#8B4513', 'fear': '#6A0DAD',
    'happy': '#FFD700', 'neutral': '#808080', 'sad': '#4169E1', 'surprise': '#FF1493'
}

# Local storage directory for selfies
SELFIES_DIR = "mood_selfies"


# Initialize local storage
def init_storage():
    """Initialize local storage directory for selfies"""
    if not os.path.exists(SELFIES_DIR):
        os.makedirs(SELFIES_DIR)

    # Create mood_data.json if it doesn't exist
    if not os.path.exists(os.path.join(SELFIES_DIR, "mood_data.json")):
        with open(os.path.join(SELFIES_DIR, "mood_data.json"), 'w') as f:
            json.dump({}, f)


def load_mood_data():
    """Load mood data from local JSON file"""
    try:
        with open(os.path.join(SELFIES_DIR, "mood_data.json"), 'r') as f:
            return json.load(f)
    except:
        return {}


def save_mood_data(mood_data):
    """Save mood data to local JSON file"""
    with open(os.path.join(SELFIES_DIR, "mood_data.json"), 'w') as f:
        json.dump(mood_data, f, indent=2)


def save_selfie_locally(image, date, emotion, confidence):
    """Save selfie image locally and update mood data"""
    init_storage()

    # Generate filename
    date_str = date.strftime("%Y-%m-%d")
    filename = f"selfie_{date_str}.jpg"
    filepath = os.path.join(SELFIES_DIR, filename)

    # Save image
    image.save(filepath, "JPEG", quality=95)

    # Update mood data
    mood_data = load_mood_data()
    mood_data[date_str] = {
        'emotion': emotion,
        'confidence': confidence.item(),
        'timestamp': datetime.datetime.now().isoformat(),
        'filename': filename,
        'mood_note': '',  # For user notes
        'mood_color': EMOTION_COLORS[emotion]  # Color coding
    }
    save_mood_data(mood_data)

    return True


def get_selfie_for_date(date):
    """Get selfie and mood data for a specific date"""
    date_str = date.strftime("%Y-%m-%d")
    mood_data = load_mood_data()

    if date_str in mood_data:
        entry = mood_data[date_str]
        filepath = os.path.join(SELFIES_DIR, entry['filename'])

        if os.path.exists(filepath):
            try:
                image = Image.open(filepath)
                return image, entry
            except:
                return None, None

    return None, None


# Load model function with caching
@st.cache_resource
def load_emotion_model():
    """Load your trained emotion detection model"""
    try:
        model = EmotionCNN(num_classes=7)
        model.load_state_dict(torch.load('assets/emotion_cnn.pth', map_location=torch.device('cpu')))
        model.eval()
        st.sidebar.success("✅ PyTorch model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image):
    """Preprocess the image for PyTorch model prediction"""
    try:
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        target_size = (48, 48)
        img_array = cv2.resize(img_array, target_size)
        img_array = img_array.astype('float32') / 255.0

        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

        return img_tensor

    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None


def predict_emotion(model, image_tensor):
    """Make prediction using the loaded PyTorch model"""
    try:
        with torch.no_grad():
            output = model(image_tensor)
            predictions = F.softmax(output, dim=1).squeeze(0).numpy()
            return predictions
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None


def create_modern_calendar(year, month):
    """Create a modern calendar view like the reference image"""
    cal = calendar.monthcalendar(year, month)
    month_name = calendar.month_name[month]
    mood_data = load_mood_data()

    # Calendar header with navigation
    st.markdown("""
    <style>
    .calendar-container {
        background: white;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .calendar-header {
        text-align: center;
        font-size: 28px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    .calendar-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    .nav-button {
        background: none;
        border: none;
        font-size: 24px;
        cursor: pointer;
        padding: 10px;
        border-radius: 50%;
        transition: background-color 0.3s;
    }
    .nav-button:hover {
        background-color: #f8f9fa;
    }
    .calendar-grid {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        gap: 15px;
        margin-top: 20px;
    }
    .day-header {
        text-align: center;
        font-weight: 600;
        color: #6c757d;
        padding: 15px 0;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .day-cell {
        aspect-ratio: 1;
        border: 2px solid #e9ecef;
        border-radius: 15px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        background: white;
        position: relative;
        min-height: 80px;
    }
    .day-cell:hover {
        border-color: #007bff;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .day-number {
        font-size: 16px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    .mood-indicator {
        font-size: 20px;
        margin-top: 5px;
    }
    .has-mood {
        border-color: #28a745;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .current-day {
        border-color: #007bff;
        border-width: 3px;
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
    }
    .current-day .day-number {
        color: white;
    }
    .empty-cell {
        border: none;
        background: none;
    }
    </style>
    """, unsafe_allow_html=True)

    # Calendar container
    st.markdown('<div class="calendar-container">', unsafe_allow_html=True)

    # Navigation and header
    col_nav1, col_header, col_nav2 = st.columns([1, 6, 1])

    with col_nav1:
        prev_month = st.button("←", key="prev_month", help="Previous month")

    with col_header:
        st.markdown(f'<div class="calendar-header">{month_name} {year}</div>', unsafe_allow_html=True)

    with col_nav2:
        next_month = st.button("→", key="next_month", help="Next month")

    # Handle navigation
    if prev_month:
        if month == 1:
            st.session_state.calendar_month = 12
            st.session_state.calendar_year = year - 1
        else:
            st.session_state.calendar_month = month - 1
        st.rerun()

    if next_month:
        if month == 12:
            st.session_state.calendar_month = 1
            st.session_state.calendar_year = year + 1
        else:
            st.session_state.calendar_month = month + 1
        st.rerun()

    # Day headers
    days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    day_cols = st.columns(7)
    for i, day in enumerate(days):
        with day_cols[i]:
            st.markdown(f'<div class="day-header">{day}</div>', unsafe_allow_html=True)

    # Calendar grid
    today = datetime.date.today()

    for week in cal:
        week_cols = st.columns(7)
        for i, day in enumerate(week):
            with week_cols[i]:
                if day == 0:
                    st.markdown('<div class="day-cell empty-cell"></div>', unsafe_allow_html=True)
                else:
                    current_date = datetime.date(year, month, day)
                    date_str = current_date.strftime("%Y-%m-%d")

                    # Check if this day has mood data
                    has_mood = date_str in mood_data
                    is_today = current_date == today

                    # Determine cell class
                    cell_class = "day-cell"
                    if is_today:
                        cell_class += " current-day"
                    elif has_mood:
                        cell_class += " has-mood"

                    # Create cell content
                    cell_content = f'<div class="{cell_class}">'
                    cell_content += f'<div class="day-number">{day}</div>'

                    if has_mood:
                        emotion = mood_data[date_str]['emotion']
                        emoji = EMOTION_EMOJIS[emotion]
                        cell_content += f'<div class="mood-indicator">{emoji}</div>'

                    cell_content += '</div>'

                    # Display cell with click functionality
                    if st.button(f"{day}", key=f"day_{day}", help=f"View {current_date.strftime('%B %d, %Y')}"):
                        st.session_state.selected_date = current_date
                        st.session_state.show_date_detail = True

                    st.markdown(cell_content, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def display_date_detail():
    """Display detailed view for selected date"""
    if 'selected_date' not in st.session_state or not st.session_state.get('show_date_detail', False):
        return

    selected_date = st.session_state.selected_date
    image, mood_entry = get_selfie_for_date(selected_date)

    st.markdown("---")

    # Create header with date and close button
    col_header, col_close = st.columns([4, 1])
    with col_header:
        st.markdown(f"### 📅 {selected_date.strftime('%B %d, %Y')}")
    with col_close:
        if st.button("✖️ Close", key="close_detail"):
            st.session_state.show_date_detail = False
            st.rerun()

    if image and mood_entry:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📸 Your Selfie")
            st.image(image, caption="Your selfie for this day", use_column_width=True)

        with col2:
            st.subheader("🎯 Mood Analysis")
            emotion = mood_entry['emotion']
            confidence = mood_entry['confidence']
            emoji = EMOTION_EMOJIS[emotion]
            color = EMOTION_COLORS[emotion]

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color} 0%, {color}80 100%); 
                        padding: 20px; border-radius: 15px; color: white; text-align: center;">
                <h2 style="margin: 0;">{emoji} {emotion.title()}</h2>
                <h3 style="margin: 5px 0;">Confidence: {confidence:.1%}</h3>
                <p style="margin: 5px 0;">Taken: {datetime.datetime.fromisoformat(mood_entry['timestamp']).strftime('%I:%M %p')}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Show mood note if exists
            # Show mood note if exists
            if 'mood_note' in mood_entry and mood_entry['mood_note']:
                st.subheader("💭 Your Note")
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
                           border-left: 4px solid {color}; margin: 10px 0; color: black;">
                    <i>"{mood_entry['mood_note']}"</i>
                </div>
                """, unsafe_allow_html=True)

            if st.button("🗑️ Delete this entry", key="delete_entry", type="secondary"):
                # Delete the image file and mood data
                mood_data = load_mood_data()
                date_str = selected_date.strftime("%Y-%m-%d")

                if date_str in mood_data:
                    # Remove from file system
                    if 'filename' in mood_data[date_str]:
                        filename = mood_data[date_str]['filename']
                        filepath = os.path.join(SELFIES_DIR, filename)
                        if os.path.exists(filepath):
                            os.remove(filepath)

                    # Remove from mood data
                    del mood_data[date_str]
                    save_mood_data(mood_data)

                    # Remove from session cache
                    if hasattr(st.session_state, 'mood_data_cache') and date_str in st.session_state.mood_data_cache:
                        del st.session_state.mood_data_cache[date_str]

                    st.success("✅ Entry deleted!")
                    st.session_state.show_date_detail = False
                    time.sleep(1)
                    st.rerun()
    else:
        st.info("📝 No selfie taken for this date yet. Take a selfie to track your mood!")

        # Option to take selfie for this date
        st.markdown("### 📷 Take a selfie for this date")
        camera_photo = st.camera_input("Take a selfie", key=f"camera_for_{selected_date}")

        if camera_photo is not None:
            try:
                image = Image.open(camera_photo)

                # Load model for analysis
                model = load_emotion_model()
                if model:
                    if st.button("🔍 Analyze & Save", key="analyze_date_selfie", type="primary"):
                        with st.spinner("Analyzing your mood..."):
                            processed_image = preprocess_image(image)

                            if processed_image is not None:
                                predictions = predict_emotion(model, processed_image)

                                if predictions is not None:
                                    predicted_class = np.argmax(predictions)
                                    predicted_emotion = EMOTION_LABELS[predicted_class]
                                    confidence = predictions[predicted_class]

                                    # Save the selfie
                                    success = save_selfie_locally(image, selected_date, predicted_emotion, confidence)
                                    if success:
                                        st.success(f"✅ Selfie saved for {selected_date.strftime('%B %d, %Y')}!")
                                        time.sleep(2)
                                        st.rerun()
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")


def display_results_with_save(predictions, image, model):
    """Display prediction results with save to calendar option"""
    if predictions is None:
        return

    predicted_class = np.argmax(predictions)
    predicted_emotion = EMOTION_LABELS[predicted_class]
    confidence = predictions[predicted_class]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Your Selfie")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("🎯 Emotion Analysis")

        # Main prediction display
        emoji = EMOTION_EMOJIS[predicted_emotion]
        color = EMOTION_COLORS[predicted_emotion]

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color} 0%, {color}80 100%); 
                    padding: 20px; border-radius: 15px; color: white; text-align: center; margin: 10px 0;">
            <h2 style="margin: 0;">{emoji} {predicted_emotion.title()}</h2>
            <h3 style="margin: 5px 0;">Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Save to calendar section
        st.subheader("💾 Save to Calendar")
        save_date = st.date_input("Select date:", value=datetime.date.today(), key="save_date")
        # Add mood note
        mood_note = st.text_area("💭 Add a mood note (optional):",
                                 placeholder="How are you feeling? What happened today?",
                                 max_chars=200)

        if st.button("📅 Save Selfie to Calendar", type="primary", use_container_width=True):
            try:
                success = save_selfie_locally(image, save_date, predicted_emotion, confidence)
                if success and mood_note.strip():
                    # Update with mood note
                    mood_data = load_mood_data()
                    date_str = save_date.strftime("%Y-%m-%d")
                    if date_str in mood_data:
                        mood_data[date_str]['mood_note'] = mood_note.strip()
                        save_mood_data(mood_data)

                st.success(f"✅ Selfie saved for {save_date.strftime('%B %d, %Y')}!")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"Error saving selfie: {str(e)}")

        # Show all emotion scores
        st.subheader("📊 All Emotion Scores")
        for emotion, score in zip(EMOTION_LABELS, predictions):
            st.progress(float(score), text=f"{EMOTION_EMOJIS[emotion]} {emotion.title()}: {score:.1%}")


def create_analytics_dashboard():
    """Create analytics dashboard"""
    # Load mood data with session cache merged
    mood_data = load_mood_data()

    # Merge with session cache for immediate updates
    if hasattr(st.session_state, 'mood_data_cache'):
        mood_data.update(st.session_state.mood_data_cache)

    if not mood_data:
        st.info("📊 No mood data available yet. Start taking daily selfies!")
        return

    # Prepare data
    dates = []
    emotions = []
    confidences = []

    for date_str, entry in mood_data.items():
        try:
            dates.append(datetime.datetime.strptime(date_str, "%Y-%m-%d").date())
            emotions.append(entry['emotion'])
            # Ensure confidence is a Python float
            confidence = entry['confidence']
            if hasattr(confidence, 'item'):
                confidence = confidence.item()
            confidences.append(float(confidence))
        except Exception as e:
            continue  # Skip corrupted entries

    if not dates:
        st.info("📊 No valid mood data available.")
        return

    df = pd.DataFrame({
        'Date': dates,
        'Emotion': emotions,
        'Confidence': confidences
    })
    df = df.sort_values('Date')

    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Entries", len(mood_data))

    with col2:
        most_common = df['Emotion'].mode()[0] if not df.empty else "None"
        st.metric("Most Common", f"{EMOTION_EMOJIS.get(most_common, '')} {most_common.title()}")

    with col3:
        avg_confidence = df['Confidence'].mean() if not df.empty else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")

    with col4:
        days_tracked = len(df)
        st.metric("Days Tracked", days_tracked)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Mood Timeline")
        if not df.empty:
            fig = px.line(df, x='Date', y='Emotion',
                          color='Emotion',
                          color_discrete_map=EMOTION_COLORS,
                          title="Your Mood Journey")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🥧 Emotion Distribution")
        if not df.empty:
            emotion_counts = df['Emotion'].value_counts()
            fig = px.pie(values=emotion_counts.values,
                         names=emotion_counts.index,
                         color=emotion_counts.index,
                         color_discrete_map=EMOTION_COLORS)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def create_digital_album():
    """Create a digital album view of all selfies organized by date"""
    st.subheader("📱 Digital Selfie Album")

    mood_data = load_mood_data()
    if not mood_data:
        st.info("📸 No selfies in your album yet. Start taking daily selfies!")
        return

    # Sort by date (newest first)
    sorted_entries = sorted(mood_data.items(), key=lambda x: x[0], reverse=True)

    # Display options
    col1, col2 = st.columns([3, 1])
    with col1:
        view_mode = st.selectbox("View Mode:", ["Grid View", "List View"])
    with col2:
        items_per_page = st.selectbox("Items per page:", [6, 12, 24], index=1)

    # Pagination
    total_items = len(sorted_entries)
    total_pages = (total_items + items_per_page - 1) // items_per_page

    if total_pages > 1:
        page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1) - 1
    else:
        page = 0

    start_idx = page * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    page_entries = sorted_entries[start_idx:end_idx]

    if view_mode == "Grid View":
        # Grid layout (3 columns)
        cols_per_row = 3
        for i in range(0, len(page_entries), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(page_entries):
                    date_str, entry = page_entries[idx]
                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

                    with cols[j]:
                        # Load and display image
                        filepath = os.path.join(SELFIES_DIR, entry['filename'])
                        if os.path.exists(filepath):
                            image = Image.open(filepath)
                            st.image(image, caption=f"{date_obj.strftime('%b %d, %Y')}")

                            emotion = entry['emotion']
                            emoji = EMOTION_EMOJIS[emotion]
                            color = EMOTION_COLORS[emotion]

                            st.markdown(f"""
                            <div style="background-color: {color}; padding: 8px; border-radius: 8px; 
                                       color: white; text-align: center; margin: 5px 0; font-size: 14px;">
                                {emoji} {emotion.title()} ({entry['confidence']:.1%})
                            </div>
                            """, unsafe_allow_html=True)

    else:  # List View
        for date_str, entry in page_entries:
            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

            with st.container():
                col1, col2 = st.columns([1, 3])

                with col1:
                    filepath = os.path.join(SELFIES_DIR, entry['filename'])
                    if os.path.exists(filepath):
                        image = Image.open(filepath)
                        st.image(image, width=150)

                with col2:
                    st.subheader(f"📅 {date_obj.strftime('%B %d, %Y')}")

                    emotion = entry['emotion']
                    emoji = EMOTION_EMOJIS[emotion]
                    color = EMOTION_COLORS[emotion]

                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 10px; border-radius: 10px; 
                               color: white; display: inline-block; margin: 5px 0;">
                        {emoji} {emotion.title()} - Confidence: {entry['confidence']:.1%}
                    </div>
                    """, unsafe_allow_html=True)

                    st.caption(f"Taken: {datetime.datetime.fromisoformat(entry['timestamp']).strftime('%I:%M %p')}")

            st.divider()

    # Show pagination info
    if total_pages > 1:
        st.info(f"Showing {start_idx + 1}-{end_idx} of {total_items} selfies (Page {page + 1} of {total_pages})")


def main():
    # Page configuration
    st.set_page_config(
        page_title="Mood Snap ",
        page_icon="📅",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize calendar state
    if 'calendar_month' not in st.session_state:
        st.session_state.calendar_month = datetime.date.today().month
    if 'calendar_year' not in st.session_state:
        st.session_state.calendar_year = datetime.date.today().year

    # Initialize storage
    init_storage()

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .nav-tabs {
        border-bottom: 2px solid #e9ecef;
        margin-bottom: 2rem;
    }
    .camera-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">📅 Mood Snap</h1>', unsafe_allow_html=True)

    # Load model
    model = load_emotion_model()

    if model is None:
        st.error("❗ Emotion model not found! Please check your model file.")
        return

    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📱 Daily Selfie", "📅 Calendar", "📱 Album", "📊 Insights", "ℹ️ About"])

    with tab1:
        st.markdown('<div class="camera-section">', unsafe_allow_html=True)
        st.subheader("📷 Take Your Daily Mood Selfie")
        st.markdown("*Capture your mood and save it to your personal calendar*")
        st.markdown('</div>', unsafe_allow_html=True)

        # Camera input
        camera_photo = st.camera_input("📸 Take a selfie")

        # Check if a photo has been taken
        if camera_photo is not None:
            try:
                # Open the image
                image = Image.open(camera_photo)

                # Display the results with the "Save to Calendar" button immediately
                with st.spinner("Analyzing your mood..."):
                    processed_image = preprocess_image(image)

                    if processed_image is not None:
                        predictions = predict_emotion(model, processed_image)

                        if predictions is not None:
                            # Directly display the results and the "Save" button
                            display_results_with_save(predictions, image, model)

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    with tab2:
        st.subheader("📅 Your Mood Calendar")

        # Display calendar
        create_modern_calendar(st.session_state.calendar_year, st.session_state.calendar_month)

        # Display date detail if selected
        display_date_detail()

        # Show recent entries
        mood_data = load_mood_data()
        if mood_data:
            st.subheader("📝 Recent Entries")
            recent_entries = sorted(mood_data.items(), key=lambda x: x[0], reverse=True)[:5]

            for date_str, entry in recent_entries:
                col1, col2, col3 = st.columns([2, 3, 1])
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()

                with col1:
                    st.write(f"**{date_obj.strftime('%b %d, %Y')}**")

                with col2:
                    emoji = EMOTION_EMOJIS[entry['emotion']]
                    st.write(f"{emoji} {entry['emotion'].title()} ({entry['confidence']:.1%})")

                with col3:
                    if st.button("👁️", key=f"view_{date_str}", help="View details"):
                        st.session_state.selected_date = date_obj
                        st.session_state.show_date_detail = True
                        st.rerun()

    with tab3:
        create_digital_album()

    with tab4:
        st.subheader("📊 Mood Analytics & Insights")
        create_analytics_dashboard()

    with tab5:
        st.subheader("ℹ️ About Mood Calendar")
        st.markdown("""
        ### 🎯 Track Your Daily Emotions

        This app helps you build a comprehensive mood calendar using daily selfies and AI-powered emotion detection.

        **Features:**
        - 📷 **Daily Selfies**: Take a selfie each day to capture your mood
        - 🤖 **AI Analysis**: Automatically detect emotions from facial expressions
        - 📅 **Visual Calendar**: See your mood history in a beautiful calendar format
        - 📊 **Analytics**: Track patterns and trends in your emotional well-being
        - 💾 **Local Storage**: All your selfies are saved locally on your device

        **Supported Emotions:**
        """)

        # Display emotion legend
        cols = st.columns(4)
        for i, (emotion, emoji) in enumerate(EMOTION_EMOJIS.items()):
            col_idx = i % 4
            with cols[col_idx]:
                color = EMOTION_COLORS[emotion]
                st.markdown(f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 10px; 
                           color: white; text-align: center; margin: 5px 0;">
                    {emoji} {emotion.title()}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        ### 📱 How to Use:
        1. **Take Daily Selfies**: Use the "Daily Selfie" tab to capture your mood
        2. **View Calendar**: Check the "Calendar" tab to see your mood history
        3. **Analyze Trends**: Use "Insights" to understand your emotional patterns
        4. **Click on Dates**: Click any date in the calendar to view your selfie and mood for that day

        ### 💡 Tips:
        - Take selfies at the same time each day for consistency
        - Ensure good lighting for better emotion detection
        - Be natural and let your genuine emotions show
        """)


if __name__ == "__main__":
    main()
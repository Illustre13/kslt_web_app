# -*- encoding: utf-8 -*-

########################################### Imports ###########################################
import os
import cv2 
import yaml
import numpy as np
from   sys import exit
import mediapipe as mp
from datetime import datetime
from flask_mysqldb import MySQL
from apps import create_app, db
from   flask_minify  import Minify
from flask_socketio import SocketIO
from   flask_migrate import Migrate
from apps.config import config_dict
from api_generator.commands import gen_api
from tensorflow.keras.models import load_model
from flask import request, jsonify, Response, session, request, jsonify, redirect





####################################################################################################

############################################### App Configuration ##################################

# WARNING: Don't run with debug turned on in production!
DEBUG = (os.getenv('DEBUG', 'False') == 'True')

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:

    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)

socketio = SocketIO(app) ###########################################################################
Migrate(app, db)

app.config['VIDEO_UPLOADS'] = 'uploads/videos'
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, app.config['VIDEO_UPLOADS'])

#######################################################################################################################

#######################################################################################################################
#############################                   MYSQL DB            ###################################################

# Configure db
with open('db.yaml', 'r') as f:
    db = yaml.load(f, Loader=yaml.SafeLoader)
app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']

mysql = MySQL(app)

########################################################################################################################
################################# Contribute upload  Video and a Kinyarwada word into the system #######################

# Fetch the Labels
@app.route('/fetch_available_labels')
def fetch_available_labels():
    cur = mysql.connection.cursor()
    resultValues = cur.execute("SELECT available_label FROM available_label")
    if resultValues > 0:
        availableLabels = cur.fetchall()
        session['available_labels'] = availableLabels
    labelCount = cur.rowcount
    response = {
        'labelCount': labelCount,
        'availableLabels': availableLabels
    }
    cur.close()
    return jsonify(response)


# Upload Video  and the word as a contribution 
@app.route('/contribute', methods=['POST'])
def upload_video():
    video = request.files.get('video')
    kinyarwanda_word = request.form.get('input_word')

    # Save the uploaded video to the server
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = f"{kinyarwanda_word}_{timestamp}.mp4"
    video_path = 'onprogress_videos/' + video_name
    video.save(video_path)

     # Connect to the MySQL database
    cur = mysql.connection.cursor()

    # Insert the video and text into the "videos" table
    sql = "INSERT INTO onprogress_label (onprogress_label, onprogress_video) VALUES (%s, %s)"
    val = (kinyarwanda_word, video_path)
    cur.execute(sql, val)
    mysql.connection.commit()

    # Redirect the user back to the same page
    return redirect('/contribute')


#######################################################################################################################
##############################                Live Translation       ##################################################


# Declarations and initializations 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
model = load_model('model\prototype_weight_final.h5')
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
finishLiveTranslation = "False"
sentence = []

# Initialize the camera and Sentence
camera = cv2.VideoCapture(0)

#later to be fetched from the database
actions = np.array(['---','yego', 'neza', 'bibi','urakoze','isibo','umurenge','igihugu','umujyi wa kigali','kicukiro','nyarugenge','akarere','muraho','amakuru','imodoka','moto','akazi','bayi','mwaramutse','papa','mama','oya','imyaka'])

# Essential Functions: Feature Extractions, Media Pide Drawings
def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(0,250,0), thickness=1, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=2), 
                             mp_drawing.DrawingSpec(color=(0,250,0), thickness=1, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh, lh, rh, lh, rh, lh, rh, lh, rh, lh, rh, lh, rh, lh, rh, lh, rh, lh, rh, lh, rh]) # magnifiying the hands value for better accuracy
   
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def gen_frames():
    # 1. New detection variables
    temp_sentence = []
    sequence = []
    global sentence
    predictions = []
    threshold = 0.95

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        while camera.isOpened():

            # Read feed
            ret, frame = camera.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                temp_sentence.append(actions[np.argmax(res)])  # displaying  on Frame
                        
                        else:
                            sentence.append(actions[np.argmax(res)])
                            temp_sentence.append(actions[np.argmax(res)])
   
                if len(temp_sentence) > 8: 
                    temp_sentence = temp_sentence[-8:]

            # Draw the sentence on the frame
            cv2.rectangle(image, (0,0), (frame_width, 40), (211, 211, 211, 128), -1)
            cv2.putText(image, ' '.join(temp_sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        camera.release()
        cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    global camera
    global finishLiveTranslation
    finishLiveTranslation = "False"
    camera = cv2.VideoCapture(0)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    camera.release()
    global finishLiveTranslation
    finishLiveTranslation = "True"
    return 'Camera stopped'

@app.route('/get_sentence')
def get_sentence():
    global sentence
    return '*'.join(sentence)  # convert the list of words to a single string separated by spaces

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global sentence
    sentence = []
    return 'OK'


@app.route('/live_translation_status')
def get_live_translation_status():
    global finishLiveTranslation
    return finishLiveTranslation  # convert the list of words to a single string separated by spaces

#############################################################################################################################
################################################### Translate  Recorded Video Feed ##########################################


#Declarations
rSentence  = []
finishRTranslation = "False"

def gen_frames_recorded_videos(video_path):
    # 1. New detection variables
    temp_sentence= []
    sequence = []
    global rSentence
    global finishRTranslation
    predictions = []
    threshold = 0.95

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        while cap.isOpened():
            # Read a frame from the video
            ret, frame = cap.read()
            
            if not ret:
                # End of video
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                # 3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        
                        if len(rSentence) > 0: 
                            if actions[np.argmax(res)] != rSentence[-1]:
                                rSentence.append(actions[np.argmax(res)])
                                temp_sentence.append(actions[np.argmax(res)])  # displaying  on Frame
                        
                        else:
                            rSentence.append(actions[np.argmax(res)])
                            temp_sentence.append(actions[np.argmax(res)])
   
                if len(temp_sentence) > 8: 
                    temp_sentence = temp_sentence[-8:]

            # Draw the sentence on the frame
            cv2.rectangle(image, (0,0), (frame_width, 40), (211, 211, 211, 128), -1)
            cv2.putText(image, ' '.join(temp_sentence), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
            
            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            # Yield the frame to the caller
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Release the video capture and destroy all windows
        cap.release()
        cv2.destroyAllWindows()
        finishRTranslation = "True"
       
           




@app.route('/rVideo_feed', methods=['POST'])
def rVideo_feed():
    global finishRTranslation
   
   
    if 'video' not in request.files:
        return jsonify({'message': 'No video file provided'}), 400

    # Get the uploaded file from the request
    video = request.files.get('video')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if video.filename == '':
        return jsonify({'message': 'No video file provided'}), 400

    if video:
        video_name = f"video_{timestamp}.mp4"
        print(os.path.join(app.config['VIDEO_UPLOADS'], video_name))
        video.save(os.path.join(app.config['VIDEO_UPLOADS'], video_name))  

        finishRTranslation = "False"

        return jsonify({'video_name':video_name}), 200
    




@app.route('/uploads/videos/<filename>')
def download_video(filename): 
    video_path = 'D:/TFOD/ksltwebapp/ksltwebapp/uploads/videos/'+filename
    return Response(gen_frames_recorded_videos(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_rSentence')
def get_rRentence():
    global rSentence
    return '*'.join(rSentence)  # convert the list of words to a single string separated by spaces

@app.route('/clear_rSentence', methods=['POST'])
def clear_rSentence():
    global rSentence
    rSentence = []
    return 'OK'


@app.route('/recorded_translation_status')
def get_recorded_translation_status():
    global finishRTranslation
    return finishRTranslation  # convert the list of words to a single string separated by spaces


###############################################################************************#######################################



if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)
    
if DEBUG:
    app.logger.info('DEBUG            = ' + str(DEBUG)             )
    app.logger.info('Page Compression = ' + 'FALSE' if DEBUG else 'TRUE' )
    app.logger.info('DBMS             = ' + app_config.SQLALCHEMY_DATABASE_URI)
    app.logger.info('ASSETS_ROOT      = ' + app_config.ASSETS_ROOT )

for command in [gen_api, ]:
    app.cli.add_command(command)
    
if __name__ == "__main__":
    socketio.run(app)

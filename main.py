from person_counter import PersonCounter
from frame_reader import LatestFrameReader

import cv2
import logging
from collections import Counter
import time
import numpy as np


if __name__ == '__main__':
    cam_url = ''
    checkpoint_path = './weights/yolov10n.pt'
    counting_time = 60 # second
    logging.basicConfig(level=logging.INFO)

    frame_reader = LatestFrameReader(cam_url)
    person_counter = PersonCounter(checkpoint_path)
    
    counted_person_list = list()
    end = time.time() + counting_time

    try:
        while time.time() <= end:
            ok, frame = frame_reader.read()
            if not ok:
                logging.error('Cannot read frame from camera')
                continue

            num_person, person_bboxes = person_counter.count(frame)
            counted_person_list.append(num_person)

            for bbox in person_bboxes:
                # if bbox is not None:
                    bbox = bbox.astype(np.int32)
                    top_left = (bbox[0], bbox[1])
                    cv2.rectangle(frame, top_left, (bbox[2], bbox[3]), color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, 'Person', (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                                        thickness=2)
            cv2.putText(frame, f'Number of person: {num_person}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 2)
            cv2.imshow('Person counter', cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2)))
            cv2.waitKey(1)
    finally:
        frame_reader.release()

    majority_voter = Counter(counted_person_list)
    major_voting_result = majority_voter.most_common(1)[0][0]
    logging.info(f'Number of person in camera: {major_voting_result}')
    logging.info(f'Number of read frames: {len(counted_person_list)}')

    cv2.destroyAllWindows()
    
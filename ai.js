// const facemesh = require('@tensorflow-models/facemesh')
const isVideo = true;
let webcamElement, webcam, model, video, predictions
let sw = true;
// require('@tensorflow/tfjs-backend-webgl');

// const facemesh = require('@tensorflow-models/facemesh');
// const backend = require('@tensorflow/tfjs-backend-webgl');
video = document.querySelector("video");
video.onloadeddata = (event) => {
    console.log('Yay! The readyState just increased to  ' +
        'HAVE_CURRENT_DATA or greater for the first time.');
    window.requestAnimationFrame(loop);
    main()
};
async function init() {
    // Convenience function to setup a webcam
    webcamElement = document.getElementById('webcam');
    webcam = new Webcam(webcamElement, 'user');
    // Load the MediaPipe Facemesh package.
    model = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.mediapipeFacemesh);
    webcam.start()
        .then(result => {
            console.log("webcam started");
        })
        .catch(err => {
            console.log(err);
        });

}

async function loop() {
    // main()
    window.requestAnimationFrame(loop);
}

async function main() {

    // Pass in a video stream (or an image, canvas, or 3D tensor) to obtain an
    // array of detected faces from the MediaPipe graph. If passing in a video
    // stream, a single prediction per frame will be returned.

    predictions = await model.estimateFaces({
        input: video
    });

    if (predictions.length > 0) {
        /*
        `predictions` is an array of objects describing each detected face, for example:

        [
          {
            faceInViewConfidence: 1, // The probability of a face being present.
            boundingBox: { // The bounding box surrounding the face.
              topLeft: [232.28, 145.26],
              bottomRight: [449.75, 308.36],
            },
            mesh: [ // The 3D coordinates of each facial landmark.
              [92.07, 119.49, -17.54],
              [91.97, 102.52, -30.54],
              ...
            ],
            scaledMesh: [ // The 3D coordinates of each facial landmark, normalized.
              [322.32, 297.58, -17.54],
              [322.18, 263.95, -30.54]
            ],
            annotations: { // Semantic groupings of the `scaledMesh` coordinates.
              silhouette: [
                [326.19, 124.72, -3.82],
                [351.06, 126.30, -3.00],
                ...
              ],
              ...
            }
          }
        ]
        */

        for (let i = 0; i < predictions.length; i++) {
            const keypoints = predictions[i].scaledMesh;

            // Log facial keypoints.
            for (let i = 0; i < keypoints.length; i++) {
                const [x, y, z] = keypoints[i];

                console.log(`Keypoint ${i}: [${x}, ${y}, ${z}]`);
            }
        }
    }
}
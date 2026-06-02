# Argus-II-Retinal-Prosthesis
end to end stimulation of the working and show case of "bionic eye" via 60 electrodes only! 
This project presents a complete end to end simulation of a retinal prosthesis (bionic eye) based on a 60 electrode array. The pipeline models the entire visual restoration process, from image acquisition and preprocessing to electrode stimulation and phosphene generation. The system demonstrates how visual information is transformed into electrical stimulation patterns and subsequently perceived as phosphene-based vision, providing an interactive showcase of the capabilities and limitations of a 60-electrode retinal implant.

This was possible via the Arduino UNO Q board,Tiny ML kit.

Hardware Deployment and Real-Time Signal Generation

-To enable real time operations, the stimulation model was deployed on an Arduino UNO Q board along with the TinyML Kit. The trained model was optimized and embedded onto the microcontroller, allowing efficient on device inference with minimal latency.
-This setup enables faster generation of electrode stimulation signals without relying on external computing resources, demonstrating the feasibility of lightweight edge-AI deployment for retinal prosthesis applications which is quite necessary!(for such kind of devices)
-[for real time data,signal trasferring to the electrodes implanted in the retina]


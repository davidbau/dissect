# Understanding the Role of Individual Units in a Deep Network

When a deep network is trained on a high-level task such as classifying a place or synthesizing a scene, individual neural units within the network will often emerge that match specific human-interpretable concepts, like "trees", "windows", or "human faces." This is striking when the matched classes are not explicitly labeled in training. Why are these units learned, and what are they for?

What role do such individual units serve within a deep network?

We examine this question in two types of networks that contain interpretable units: networks trained to classify images of scenes (supervised image classifiers), and networks trained to synthesize images of scenes (generative adversarial networks).

## Dissecting Units in Classifiers and Generators

Network dissection compares individual network units to the predictions of a semantic segmentation network that has been trained to label pixels with a broad set of object, part, materrial and color classes. This technique gives us a standard and scalable way to identify any units within the networks we analyze that match those same semantic classes.

It works both in classification settings where the image is the input, and in generative settings where the image is the output.

![Dissection](/www/classifier-dissection.png)

We find that both state-of-the-art GANs and classifiers contain object-matching units that correspond to a variety of object and part concepts, with semantics emerging in different layers.

![Comparing a Classifier to a Generator](/www/dissection-compare.png)

To investigate the role of such units within classifiers, we measure the impact on the accuracy of the network when we turn off units individually or in groups. We find that removing as few as 20 units can destroy the network's ability to detect a class, but retaining only those 20 units and removing 492 other units in the same layer can keep the network's accuracy on that same class mostly intact. Furthermore, we find that those units that are important for the largest number of output classes are also the emergent units that match human-interpretable concepts best.

![Classifier Intervention Experiments](/www/classifier-intervention.png)

In a generative network, we can understand causal effects of neurons by observing changes to output images when sets of units are turned on and off. We find causal effects are strong enough to enable users to paint images out of object classes by activating neurons; we also find that some units reveal interactions between objects and specific contexts within a model.

![Genereator Intervention Experiments](/www/generator-intervention.png)

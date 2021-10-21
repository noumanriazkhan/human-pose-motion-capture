# Module 4:- 3D pose estimation (output of 2d pose is fed into 3D pose estimator... Can be replaced very easily... To get 3D pose animation) -- this is the moneymaker here, just need to replace this with a better one and/or train a better one, have a few shortlisted (the sliding you see in the output is happening here)

# Module 5 :- Projection Pose for camera to real world projection (Projection pose technique is used to find the camera to reak world translation in 2D) -- need to apply depth network to get accurate z-axis translation, could be also solved in module 4

# Module 6:- Normalizing and applying translation to 3D pose output. (Translation and Pose are both normalized to fit the same scale and translation is then applied to the Pose)

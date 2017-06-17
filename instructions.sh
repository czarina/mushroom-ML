sudo nvidia-docker run -it -p 8888:8888 -p 6006:6006 -v /tf_files:/tf_files --volume tf_files:/tf_files --workdir /tf_files --publish 6006:6006 tensorflow/tensorflow:latest-gpu


python retrain.py \
  --bottleneck_dir=all_bottlenecks/ \
  --model_dir=bucket_model/ \
  --summaries_dir=species_model_augmented/training_summaries/long \
  --output_graph=bucket_model/retrained_graph.pb \
  --output_labels=bucket_model/retrained_labels.txt \
  --image_dir=bucket_data_iterative \
  --how_many_training_steps=12000 \
  --learning_rate=0.025 \
  --random_brightness=10 \
  --flip-left-right=True \ 
  --random_crop=10 \
  --random_scale=10 \


  tensorboard --logdir species_model_augmented/training_summaries --host 0.0.0.0 --port 6006 &

Optimize for mobile: https://petewarden.com/2016/09/27/tensorflow-for-mobile-poets/

install Bazel: https://bazel.build/versions/master/docs/install-ubuntu.html

start new docker: sudo nvidia-docker run -it -p 8889:8889 -v /tf_files:/tf_files tensorflow/tensorflow:nightly-devel
cd /tensorflow

bazel build tensorflow/python/tools:optimize_for_inference


bazel-bin/tensorflow/python/tools/optimize_for_inference \
--input=/tf_files/species_model_augmented/retrained_graph.pb \
--output=/tf_files/species_model_augmented/optimized_graph.pb \
--input_names=Mul \
--output_names=final_result

bazel build tensorflow/tools/quantization:quantize_graph

bazel-bin/tensorflow/tools/quantization/quantize_graph \
--input=/tf_files/species_model_augmented/optimized_graph.pb \
--output=/tf_files/species_model_augmented/rounded_graph.pb \
--output_node_names=final_result \
--mode=weights_rounded

bazel build tensorflow/contrib/util:convert_graphdef_memmapped_format

bazel-bin/tensorflow/contrib/util/convert_graphdef_memmapped_format \
--in_graph=/tf_files/species_model_augmented/optimized_graph.pb \
--out_graph=/tf_files/species_model_augmented/mmapped_graph.pb


XCODE STUFF
brew install libtool

scp -i ../mushroom2.pem  ubuntu@54.215.232.58:/tf_files/species_model_augmented/mmapped_graph.pb tensorflow/examples/ios/camera/data/
scp -i ../mushroom2.pem  ubuntu@54.215.232.58:/tf_files/species_model_augmented/retrained_labels.txt tensorflow/examples/ios/camera/data/

open tensorflow/examples/ios/camera/camera_example.xcodeproj
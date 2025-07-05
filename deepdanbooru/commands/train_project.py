import os
import random
from sqlite3.dbapi2 import NotSupportedError
import time
import datetime
import numpy as np
import tensorflow as tf

import deepdanbooru as dd
import deepdanbooru.project
import deepdanbooru.data
import deepdanbooru.model
import deepdanbooru.io

def export_model_as_float32(temporary_model, checkpoint_path, export_path):
    checkpoint = tf.train.Checkpoint(model=temporary_model)
    manager = tf.train.CheckpointManager(
        checkpoint=checkpoint, directory=checkpoint_path, max_to_keep=3
    )
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    temporary_model.save(export_path, include_optimizer=False)

def train_project(project_path, source_model):
    project_context_path = os.path.join(project_path, "project.json")
    project_context = dd.io.deserialize_from_json(project_context_path)

    width = project_context["image_width"]
    height = project_context["image_height"]
    database_path = project_context["database_path"]
    minimum_tag_count = project_context["minimum_tag_count"]
    model_type = project_context["model"]
    optimizer_type = project_context["optimizer"]
    learning_rate = project_context.get("learning_rate", 0.001)
    learning_rates = project_context.get("learning_rates")
    gamma_schedule = project_context.get("gamma_schedule")
    minibatch_size = project_context["minibatch_size"]
    epoch_count = project_context["epoch_count"]
    export_model_per_epoch = project_context.get("export_model_per_epoch", 10)
    checkpoint_frequency_mb = project_context["checkpoint_frequency_mb"]
    console_logging_frequency_mb = project_context["console_logging_frequency_mb"]
    rotation_range = project_context["rotation_range"]
    scale_range = project_context["scale_range"]
    shift_range = project_context["shift_range"]
    use_mixed_precision = project_context.get("mixed_precision", False)
    loss_type = project_context.get("loss", "binary_crossentropy")
    checkpoint_path = os.path.join(project_path, "checkpoints")

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if optimizer_type == "adam":
        optimizer = tf.optimizers.Adam(learning_rate)
        print("Using Adam optimizer ... ")
    elif optimizer_type == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate, momentum=0.9, nesterov=True)
        print("Using SGD optimizer ... ")
    elif optimizer_type == "rmsprop":
        optimizer = tf.optimizers.RMSprop(learning_rate)
        print("Using RMSprop optimizer ... ")
    else:
        raise Exception(f"Not supported optimizer : {optimizer_type}")

    if use_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        print("Optimizer is changed to LossScaleOptimizer.")

    if model_type == "resnet_152":
        model_delegate = dd.model.resnet.create_resnet_152
    elif model_type == "resnet_custom_v1":
        model_delegate = dd.model.resnet.create_resnet_custom_v1
    elif model_type == "resnet_custom_v2":
        model_delegate = dd.model.resnet.create_resnet_custom_v2
    elif model_type == "resnet_custom_v3":
        model_delegate = dd.model.resnet.create_resnet_custom_v3
    elif model_type == "resnet_custom_v4":
        model_delegate = dd.model.resnet.create_resnet_custom_v4
    else:
        raise Exception(f"Not supported model : {model_type}")

    print("Loading tags ... ")
    tags = dd.project.load_tags_from_project(project_path)
    output_dim = len(tags)

    print(f"Creating model ({model_type}) ... ")

    if source_model:
        print("--- Fine-tuning Mode (Output Expansion) Activated ---")
        print(f"Loading pre-trained model from: {source_model}")
        try:
            base_model = tf.keras.models.load_model(source_model, compile=False)
        except Exception as e:
            print(f"Error loading model: {e}"); return
        
        original_output_conv = next((l for l in reversed(base_model.layers) if isinstance(l, tf.keras.layers.Conv2D)), None)
        if original_output_conv is None:
            raise ValueError("Could not find the final Conv2D layer in the base model.")
        
        OLD_TAG_COUNT = original_output_conv.filters
        new_tag_count = output_dim
        print(f"Original model tag count: {OLD_TAG_COUNT}")
        print(f"New total tag count: {new_tag_count}")

        if new_tag_count <= OLD_TAG_COUNT:
            raise ValueError(f"For expansion, new tag count ({new_tag_count}) must be greater than old tag count ({OLD_TAG_COUNT}).")
        
        use_bias_in_new_layer = len(original_output_conv.get_weights()) == 2
        feature_extractor_output = original_output_conv.input
        new_output_conv = tf.keras.layers.Conv2D(
            filters=new_tag_count, kernel_size=original_output_conv.kernel_size,
            name='expanded_conv_output', strides=original_output_conv.strides,
            padding=original_output_conv.padding, use_bias=use_bias_in_new_layer,
            kernel_initializer='zeros')(feature_extractor_output)
        x = new_output_conv
        for layer in base_model.layers[base_model.layers.index(original_output_conv) + 1:]:
            x = layer.__class__.from_config(layer.get_config())(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=x)
        
        print("Copying weights for all non-output layers...")
        for layer in model.layers:
            if layer.name == 'expanded_conv_output': continue
            try:
                base_layer = base_model.get_layer(name=layer.name)
                if base_layer.get_weights():
                    layer.set_weights(base_layer.get_weights())
            except (ValueError, IndexError): pass
        
        print("Copying and expanding weights for the output layer...")
        original_weights_list = original_output_conv.get_weights()
        if len(original_weights_list) == 2:
            old_weights, old_biases = original_weights_list
            new_weights = np.zeros(model.get_layer('expanded_conv_output').get_weights()[0].shape, dtype=np.float32)
            new_biases = np.zeros(model.get_layer('expanded_conv_output').get_weights()[1].shape, dtype=np.float32)
            new_weights[:, :, :, :OLD_TAG_COUNT] = old_weights
            new_biases[:OLD_TAG_COUNT] = old_biases
            model.get_layer('expanded_conv_output').set_weights([new_weights, new_biases])
        elif len(original_weights_list) == 1:
            old_weights = original_weights_list[0]
            new_weights = np.zeros(model.get_layer('expanded_conv_output').get_weights()[0].shape, dtype=np.float32)
            new_weights[:, :, :, :OLD_TAG_COUNT] = old_weights
            model.get_layer('expanded_conv_output').set_weights([new_weights])
        
        print("Freezing layers of the base model...")
        for layer in model.layers:
            if layer.name != 'expanded_conv_output':
                layer.trainable = False
            else:
                layer.trainable = True
        model.summary()
    else:
        print("--- Training from scratch ---")
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
        inputs = tf.keras.Input(shape=(height, width, 3), dtype=tf.float32)
        outputs = model_delegate(inputs, output_dim)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_type)
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy("float32")
            tf.keras.mixed_precision.set_global_policy(policy)
            inputs_float32 = tf.keras.Input(shape=(height, width, 3), dtype=tf.float32)
            outputs_float32 = model_delegate(inputs_float32, output_dim)
            model_float32 = tf.keras.Model(inputs=inputs_float32, outputs=outputs_float32, name=model_type)
            print("float32 model is created.")
        print(f"Model : {model.input_shape} -> {model.output_shape}")

    if loss_type == "binary_crossentropy":
        loss = tf.keras.losses.BinaryCrossentropy()
    elif loss_type == "focal_loss":
        initial_gamma = 2.0
        if gamma_schedule:
            initial_gamma = sorted(gamma_schedule, key=lambda x: x['used_epoch'])[0]['gamma']
        print(f"Initializing Focal Loss with gamma = {initial_gamma} ...")
        loss = dd.model.losses.focal_loss(gamma=initial_gamma)
    else:
        raise NotSupportedError(f"Loss type '{loss_type}' is not supported.")
    print(f"Using loss : {loss_type}")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    print(f"Loading database ... ")
    image_records = dd.data.load_image_records(database_path, minimum_tag_count)

    used_epoch = tf.Variable(0, dtype=tf.int64)
    used_minibatch = tf.Variable(0, dtype=tf.int64)
    used_sample = tf.Variable(0, dtype=tf.int64)
    offset = tf.Variable(0, dtype=tf.int64)
    random_seed = tf.Variable(0, dtype=tf.int64)
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, model=model, used_epoch=used_epoch,
        used_minibatch=used_minibatch, used_sample=used_sample,
        offset=offset, random_seed=random_seed,
    )
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_path, max_to_keep=3)

    if manager.latest_checkpoint:
        print(f"Checkpoint exists. Continuing training ... ({datetime.datetime.now()})")
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"used_epoch={int(used_epoch)}, ...")
    else:
        print(f"No checkpoint. Starting new training ... ({datetime.datetime.now()})")

    epoch_size = len(image_records)
    slice_size = minibatch_size * checkpoint_frequency_mb
    loss_sum, loss_count, used_sample_sum = 0.0, 0, 0
    last_time = time.time()

    while int(used_epoch) < epoch_count:
        current_epoch = int(used_epoch)

        if learning_rates:
            for schedule in learning_rates:
                if schedule["used_epoch"] <= current_epoch:
                    learning_rate = schedule["learning_rate"]
        print(f"Trying to change learning rate to {learning_rate} ...")
        optimizer.learning_rate.assign(learning_rate)
        tf.print(f"Learning rate is changed to", optimizer.learning_rate, "...")

        if loss_type == "focal_loss" and gamma_schedule:
            current_gamma = model.loss.gamma
            new_gamma = current_gamma
            for schedule in gamma_schedule:
                if schedule["used_epoch"] <= current_epoch:
                    new_gamma = schedule["gamma"]
            
            if new_gamma != current_gamma:
                print(f"Updating Focal Loss gamma from {current_gamma} to {new_gamma} ...")
                new_loss = dd.model.losses.focal_loss(gamma=new_gamma)
                model.compile(optimizer=optimizer, loss=new_loss, metrics=model.metrics)
                print("Model re-compiled with new gamma value.")

        print(f"Shuffling samples (epoch {current_epoch}) ... ")
        epoch_random = random.Random(int(random_seed))
        epoch_random.shuffle(image_records)

        while int(offset) < epoch_size:
            image_records_slice = image_records[int(offset) : min(int(offset) + slice_size, epoch_size)]
            image_paths = [rec[0] for rec in image_records_slice]
            tag_strings = [rec[1] for rec in image_records_slice]
            dataset_wrapper = dd.data.DatasetWrapper(
                (image_paths, tag_strings), tags, width, height,
                scale_range=scale_range, rotation_range=rotation_range, shift_range=shift_range
            )
            dataset = dataset_wrapper.get_dataset(minibatch_size)
            for (x_train, y_train) in dataset:
                sample_count = x_train.shape[0]
                step_result = model.train_on_batch(x_train, y_train, return_dict=True)
                used_minibatch.assign_add(1)
                used_sample.assign_add(sample_count)
                used_sample_sum += sample_count
                loss_sum += step_result['loss']
                loss_count += 1
                if int(used_minibatch) % console_logging_frequency_mb == 0:
                    current_time = time.time()
                    delta_time = current_time - last_time
                    p = step_result['precision']
                    r = step_result['recall']
                    f1 = 2.0 * (p * r) / (p + r) if p + r > 0.0 else 0.0
                    avg_loss = loss_sum / float(loss_count)
                    sps = float(used_sample_sum) / max(delta_time, 0.001)
                    progress = float(int(used_sample)) / float(epoch_size * epoch_count) * 100.0
                    remain_sec = float(epoch_size * epoch_count - int(used_sample)) / max(sps, 0.001)
                    eta = datetime.datetime.now() + datetime.timedelta(seconds=remain_sec)
                    print(f"Epoch[{current_epoch}] Loss={avg_loss:.6f}, P={p:.6f}, R={r:.6f}, F1={f1:.6f}, Speed = {sps:.1f} samples/s, {progress:.2f} %, ETA = {eta:%Y-%m-%d %H:%M:%S}")
                    model.reset_metrics()
                    loss_sum, loss_count, used_sample_sum = 0.0, 0, 0
                    last_time = current_time
            offset.assign_add(slice_size)
            print(f"Saving checkpoint ... ({datetime.datetime.now()})")
            manager.save()
        
        used_epoch.assign_add(1)
        random_seed.assign_add(1)
        offset.assign(0)

        if export_model_per_epoch > 0 and int(used_epoch) % export_model_per_epoch == 0:
            export_path = os.path.join(project_path, f"model-{model_type}.e{int(used_epoch)}.keras")
            model.save(export_path, include_optimizer=False)
            if use_mixed_precision and not source_model:
                export_model_as_float32(model_float32, checkpoint_path, export_path + ".float32.keras")

    print("Saving model ...")
    model_path = os.path.join(project_path, f"model-{model_type}.keras")
    model.save(model_path, include_optimizer=False)
    if use_mixed_precision and not source_model:
        export_model_as_float32(model_float32, checkpoint_path, model_path + ".float32.keras")

    print("Training is complete.")
    print(f"used_epoch={int(used_epoch)}, used_minibatch={int(used_minibatch)}, used_sample={int(used_sample)}")

from nuscenes.nuscenes import NuScenes
import matplotlib
import tempfile
import numpy as np
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

nusc = NuScenes(version='v1.0-mini',
                dataroot="/Users/shreysharma/Documents/ML@Purdue/robot_vision_onboarding/nuscenes")

# nusc.list_scenes()
# my_scene = nusc.scene[0]
#
# first_sample_token = my_scene['first_sample_token']
# # nusc.render_sample(first_sample_token)
# my_sample = nusc.get('sample', first_sample_token)
# print(my_sample)
# nusc.list_sample(my_sample['token'])
#
# sensor = 'CAM_FRONT'
# cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
# print(cam_front_data)
# # nusc.render_sample_data(cam_front_data['token'])
#
# my_annotation_token = my_sample['anns'][18]
# my_annotation_metadata = nusc.get('sample_annotation', my_annotation_token)
# nusc.explorer.render_annotation(my_annotation_token, out_path="annotation.png")
# print("Saved to annotation.png")

for i in range(len(nusc.category)):
    cat_name = nusc.category[i]["name"]
    description = nusc.category[i]["description"]
    annotations = [ann for ann in nusc.sample_annotation if ann["category_name"] == cat_name]
    if not annotations:
        continue
    attribute_examples = {}
    different_attributes = []
    different_visibilities = []
    for annotation in annotations:
        for token in annotation["attribute_tokens"]:
            attribute = nusc.get("attribute", token)["name"]
            visibility = annotation["visibility_token"]
            if attribute not in different_attributes:
                different_attributes.append(attribute)
            if visibility not in different_visibilities:
                different_visibilities.append(visibility)
            if (attribute, visibility) not in attribute_examples:
                attribute_examples[(attribute, visibility)] = [annotation["token"]]
            else:
                attribute_examples[(attribute, visibility)].append(annotation["token"])
    if not attribute_examples:
        continue

    columns = len(different_attributes)
    rows = len(different_visibilities)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 15), tight_layout=True)
    plt.suptitle(cat_name)
    keys = sorted(attribute_examples.keys(), key=lambda x: (x[1], x[0]))
    for ax, name in zip(fig.axes, keys):
        annotation_token = attribute_examples[name][0]

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        nusc.explorer.render_annotation(annotation_token, out_path=tmp.name)
        render_fig = plt.gcf()
        plt.close(render_fig)

        subtitle = name[0] + ", visibility = " + name[1]
        ax.set_title(subtitle)

        img = matplotlib.image.imread(tmp.name)
        h, w, _ = img.shape
        img = img[:, w//2:, :]
        ax.set_aspect("auto")
        ax.set_anchor("C")
        ax.imshow(img)
        ax.axis("off")

    axes = np.atleast_1d(axes).ravel()
    for ax in axes.flatten()[len(keys):]:
        ax.axis("off")
    plt.savefig(cat_name + "_attributes.png")

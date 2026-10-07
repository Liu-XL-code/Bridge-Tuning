from monai import transforms


def get_preload_transforms(args, is_train=True):
    """
    获取预加载时的变换（一次性完成的预处理步骤）
    Args:
        args: 配置参数
        is_train: 是否为训练模式。训练模式会padding到固定大小，测试模式保持原始大小
    """
    base_transforms = [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest")
        ),
        transforms.ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min, a_max=args.a_max,
            b_min=args.b_min, b_max=args.b_max,
            clip=True
        ),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

        transforms.SpatialPadd(
            keys=["image", "label"],
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
        )
    ]

    base_transforms.append(transforms.ToTensord(keys=["image", "label"]))

    return transforms.Compose(base_transforms)


def get_augmentation_transforms(args):
    """获取训练时的数据增强变换（每次训练时执行）"""
    augmentation_transform = transforms.Compose(
        [
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        ]
    )
    return augmentation_transform

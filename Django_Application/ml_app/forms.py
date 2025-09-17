from django import forms

class VideoUploadForm(forms.Form):
    upload_video_file = forms.FileField(
        label="Select Video",
        required=True,
        widget=forms.FileInput(attrs={"accept": "video/*"})
    )
    sequence_length = forms.IntegerField(
        label="Sequence Length",
        required=True,
        min_value=1,
        max_value=300,  # Example upper limit
        initial=30
    )

    def clean_upload_video_file(self):
        file = self.cleaned_data.get("upload_video_file")
        if file:
            if file.size > 50 * 1024 * 1024:  # 50 MB limit
                raise forms.ValidationError("Video file too large (max 50 MB).")
            if not file.name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                raise forms.ValidationError("Unsupported video format.")
        return file

class ImageUploadForm(forms.Form):
    upload_image_file = forms.ImageField(
        label="Select Image",
        required=True,
        widget=forms.ClearableFileInput(attrs={
            "accept": ".jpg,.jpeg,.png,.bmp,.tiff",
            "class": "dropzone",  # Add a class for Dropzone-like UX (frontend must implement)
            "multiple": False,
        }),
        error_messages={
            "invalid": "Please upload a valid image file.",
            "required": "Please select an image to upload.",
        }
    )

    def clean_upload_image_file(self):
        file = self.cleaned_data.get("upload_image_file")
        allowed_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        max_size = 10 * 1024 * 1024  # 10 MB
        if file:
            # Validate file size
            if file.size > max_size:
                raise forms.ValidationError("Image file too large (max 10 MB).")
            # Validate file extension
            filename = file.name.lower()
            if not filename.endswith(allowed_extensions):
                raise forms.ValidationError(
                    "Unsupported image format. Allowed formats: .jpg, .jpeg, .png, .bmp, .tiff"
                )
        return file
Hi,

This folder has all the newspaper pages I annotated, along with the boxes I drew and the OCR text for each one. There are 100 pages and 8,074 labelled regions in total, and every region has its transcription.

I used four labels: Headline, Subtitle, BodyText and Caption. I took out the Image boxes on purpose, so you won't find any of those.

There are two ways to use this folder.

1) Put it into your own Label Studio (you get the pages, boxes and text, and you can still edit them)

   Everything for this is in the folder "1_import_into_label_studio".

   - Make a new project in Label Studio.
   - Open Settings > Labeling Interface > Code, paste in everything from label_config.xml, and save.
   - Go to the Data Manager, click Import, and upload import_annotated_pages.json.
     Import it as tasks, not as plain files.

   The JSON file is big (about 176 MB) because the page images are stored inside it. You don't need to copy any other images. If the upload is rejected for being too large, ask me and I'll make a smaller version.

2) Just read the data, no Label Studio needed

   Look in the folder "2_readable_export":

   - all_annotations.json has every page in one file.
   - the json folder has one file per page. Each region lists its label, its box (in pixels and in percent) and its OCR text.
   - the images folder has each page as a picture with the boxes drawn on it, so you can check things quickly by eye.

One thing to keep in mind: the OCR text is the transcription I kept while annotating (Gemini helped with it). It's good for training and comparing, but it isn't perfect ground truth, so don't treat it as 100% correct.

Rajdip

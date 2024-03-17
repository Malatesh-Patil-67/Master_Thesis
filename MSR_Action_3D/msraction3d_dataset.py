class MyCustomDataset(Dataset):
    def __init__(self, root_dir, mode='train', sliding_window_size=91):
        self.root_dir = root_dir
        self.subjects = os.listdir(root_dir)
        self.frame_data = []  # Stores the information of each frame
        self.labels = [] # Stores the labels of each frame
        self.indexvalues = []
        self.joint_coordinates = []
        self.sliding_window_size = sliding_window_size
        self._process_data(mode)

    def __len__(self):
        return len(self.frame_data)



    def __getitem__(self, index):

        center_frame = self.frame_data[index]

        label = self.labels[index]

        skeleton_number = center_frame['skeleton_number']

        # Sliding window logic

        skeleton_start_index = self.indexvalues[skeleton_number]['start_index']
        skeleton_end_index = self.indexvalues[skeleton_number]['end_index']


        start_index = index - self.sliding_window_size // 2
        end_index = index + self.sliding_window_size // 2


        if (start_index < skeleton_start_index):
          start_index = skeleton_start_index

        else:
          start_index = max(start_index, 0)

        if (end_index > skeleton_end_index):
          end_index = skeleton_end_index


        t = []

        for i in range(start_index, end_index + 1):
          frame_data = self.frame_data[i]
          joint_coordinates = frame_data['joint_coordinates']
          t.append(joint_coordinates)


        while len(t) < self.sliding_window_size:
          t.append([0.0] * 60)  # Assuming 25 joints with (x, y, z) coordinates

        #if len(t) < 91 and isinstance(t[0], list):  # Check if t contains elements that are lists
            #print("Shape of t before tensor conversion:", len(t), len(t[0]))


        t = torch.tensor(t, dtype=torch.float32)
        t = t.view(self.sliding_window_size, 20, 3)


        return t, label

    def _process_data(self, mode):
        frame_counter = 0  # Counter for sequential frame numbering
        skeleton_temp=0
        indices = []
        counting = 0

        if mode == 'train':
            data_dir = os.path.join(self.root_dir, 'train')
        elif mode == 'test':
            data_dir = os.path.join(self.root_dir, 'test')
        else:
            raise ValueError("Invalid mode. Mode should be 'train' or 'test'.")



        skeleton_folder = os.listdir(data_dir)

        for skeleton_file in skeleton_folder:
          skeleton_dir = os.path.join(data_dir, skeleton_file)


          # Action name
          action_folder = skeleton_dir[-23:-21]


          # Read and process the skeleton.txt file to extract frame-level data
          frames, labels , start , end= self._read_skeleton_file(skeleton_dir, frame_counter, action_folder , skeleton_temp)
          counting = counting +1

          indices.append({
          'start_index': start,
          'end_index': end,

          })
          self.frame_data.extend(frames)
          self.labels.extend(labels)

          # Update the frame counter based on the number of frames in the current skeleton file
          frame_counter += len(frames)
          skeleton_temp = skeleton_temp + 1

        self.indexvalues.extend(indices)


    def _read_skeleton_file(self, skeleton_file, frame_counter, action , skeleton_temp):
        frames = []
        labels = []
        start_ind = frame_counter
        temp_index = 0


        # Read and parse the skeleton.txt file to extract frame-level data

        with open(skeleton_file, 'r') as f:
          fcounter = 0


          lines = f.readlines()


          line_counter = 0


          while line_counter < (len(lines) - 20):

            v = lines[line_counter].strip().split()


            frame_number = frame_counter + fcounter
            fcounter=fcounter + 1
            coordinates = []
            # Process the next 20 lines
            for i in range(20):
              # Make sure we don't go beyond the total number of lines in the file

              if line_counter < len(lines):
                current_line = lines[line_counter].strip().split()

                # Process the content of the current line here
                for k in current_line[0:3]:

                  coordinates.append(float(k))

                  self.joint_coordinates.append(coordinates)

              line_counter = line_counter + 1


            frames.append({
            'frame_number': frame_number,
            'joint_coordinates': coordinates,
            'skeleton_number' : skeleton_temp,

            })
            label = action
            labels.append(label)



        end_ind = frame_number

        temp_index = frame_number


        return frames, labels, start_ind, temp_index
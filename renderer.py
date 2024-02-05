class PyRenderer():
	"""PyRenderer."""

	def __init__(self, mano_poses, out_dir, cam_intrinsics, mano_betas, images=None, meta=None, mano_side="right", h=480, w=640, device='cuda',
				render_name="render1", video_only=True, framerate=30, verbose=False):
		"""Constructor.
		Args:
		  	mano_poses: Trajectory of mano poses.
			images: background images. (optional)
		"""
		self._mano_poses = mano_poses
		self._num_frames = self._mano_poses.shape[1]
		# if mano_betas is None:
		# 	self._mano_betas = mano_betas_default.copy()
		# else:
		# 	self._mano_betas = mano_betas

		self._images = images
		self._device = torch.device(device)
		self._video_only = video_only
		self._framerate = framerate
		self._verbose = verbose
		self._num_cameras = 1
		self._w = w
		self._h = h
		self._K = []

		# Load intrinsics.
		K = torch.tensor(
			[[cam_intrinsics['fx'], 0.0, cam_intrinsics['ppx']], 
			[0.0, cam_intrinsics['fy'], cam_intrinsics['ppy']], 
			[0.0, 0.0, 1.0]],
			dtype=torch.float32,
			device=self._device)
		self._K.append(K)

		# Create pyrender cameras.
		self._cameras = []
		for c in range(self._num_cameras):
			K = self._K[c].cpu().numpy()
			fx = K[0][0].item()
			fy = K[1][1].item()
			cx = K[0][2].item()
			cy = K[1][2].item()
			cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
			self._cameras.append(cam)

		self._mano_side = mano_side
		# self._mano_group_layer = MANOGroupLayer([self._mano_side],
		# 										[self._mano_betas.astype(np.float32)], mano_root=base_path+"/utils/mano/manopth/mano_v1_2/models").to(self._device)
		self._num_mano = 1

		self._faces = []

		self._r = pyrender.OffscreenRenderer(viewport_width=self._w, viewport_height=self._h)

		self._render_dir = [
			os.path.join(os.path.dirname(__file__), "..", out_dir[2:], "rendered",
						 render_name, str(c))
			for c in range(self._num_cameras)
		]

		self._mano_vert = []
		self._mano_joint_3d = []
		self._mano_joint = []

		verts = np.zeros((self._num_frames, 778, 3))
		joints = np.zeros((self._num_frames, 21, 3))
		# print("self._num_mano: ", self._num_mano)
		for c in range(self._num_cameras):
			for f in range(self._num_frames):
				if not mano_betas[f].any():
					self._mano_betas = mano_betas_default.copy()
				else:
					self._mano_betas = mano_betas[f].copy()

				self._mano_group_layer = MANOGroupLayer([self._mano_side],
												[self._mano_betas.astype(np.float32)], 
												mano_root=base_path+"/utils/mano/manopth/mano_v1_2/models").to(self._device)

				self._faces.append(self._mano_group_layer.f.cpu().numpy())

				mano_pose = self._mano_poses[:,f,1:]
				pose = torch.from_numpy(mano_pose.astype(np.float32)).to(self._device)
				pose = pose.view(-1, self._mano_group_layer.num_obj * 51)

				vert, joint = self._mano_group_layer(pose)
				# for debug
				# if f==10:
				# 	print("frame #%d" %f)
				# 	print("mano_betas:", self._mano_betas)
				# 	print("vert.shape:", vert.shape) 
				# 	print("joint.shape:", joint.shape) 

				verts[f, :, :] = vert.cpu().numpy()
				joints[f, :, :] = joint.cpu().numpy()

			mano_vert = [
				  np.zeros((self._num_frames, 778, 3), dtype=np.float32)
				  for _ in range(self._num_mano)
			]

			mano_joint = [
				  np.zeros((self._num_frames, 21, 3), dtype=np.float32)
				  for _ in range(self._num_mano)
			]

			for o in range(self._num_mano):
				mano_vert[o] = verts[:, 778 * o: 778 * (o + 1), :]
				mano_joint[o] = joints[:, 21 * o: 21 * (o + 1), :]
			
			self._mano_vert.append(mano_vert)
			self._mano_joint.append(mano_joint)


	def _blend(self, im_real, im_render):
		"""Blends the real and rendered images.

		Args:
		  	im_real: A uint8 numpy array of shape [H, W, 3] containing the real image.
		  	im_render: A uint8 numpy array of shape [H, W, 3] containing the rendered
				image.
		"""
		im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
		im = im.astype(np.uint8)
		return im

	def _render_color(self):
		"""Renders and saves mesh images."""
		for d in self._render_dir:
			os.makedirs(d, exist_ok=True)

		if self._verbose:
			print('Rendering mesh image')
		for c in range(self._num_cameras):

			video_writer = skvideo.io.FFmpegWriter(self._render_dir[c] + '/traj.mp4', 
				inputdict={'-r': str(self._framerate), '-s':'{}x{}'.format(self._w, self._h)},
				outputdict={'-c:v': 'libx264', '-r': str(self._framerate), '-preset': 'ultrafast',})

			print("Write video frame...")
			for i in tqdm(range(self._num_frames)):
				if self._verbose:
					print('{:03d}/{:03d}'.format(i + 1, self._num_frames))

				mano_vert = [[y[i] for y in x] for x in self._mano_vert]

				# Create pyrender scene.
				scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
									  ambient_light=np.array([1.0, 1.0, 1.0]))

				# Add camera.
				cam_pose = np.eye(4)
				scene.add(self._cameras[c], pose=cam_pose)

				vert_m = mano_vert[c]

				# Add MANO meshes.
				for o in range(self._num_mano):
					if np.all(vert_m[o] == 0.0):
						continue
					vert = vert_m[o].copy()
					vert[:, 1] *= -1
					vert[:, 2] *= -1
					mesh = trimesh.Trimesh(vertices=vert, faces=self._faces[i])
					mesh1 = pyrender.Mesh.from_trimesh(mesh)
					mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
					mesh2 = pyrender.Mesh.from_trimesh(mesh, wireframe=True)
					mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
					node1 = scene.add(mesh1)
					node2 = scene.add(mesh2)

				color, _ = self._r.render(scene)

				if self._images is not None:
					im = self._images[c][i]
					b_color = self._blend(im, color)
				else:
					b_color = color

				if not self._video_only:
					color_im_file = self._render_dir[c] + "/color_{:06d}.jpg".format(i)
					cv2.imwrite(color_im_file, b_color[:, :, ::-1])
				# video_writer.write(b_color[:, :, ::-1])
				video_writer.writeFrame(b_color[:, :, ::-1])

			#video_writer.release()
			video_writer.close()

	def run(self):
		"""Runs the renderer."""
		self._render_color()
		# self._render_joint()
	
	def close(self):
		"""Closes the renderer."""
		self._r.delete()
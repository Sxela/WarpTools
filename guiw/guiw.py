# (c) Alex Spirin 2022-2023
import os 
import json
from glob import glob
import k_diffusion as K
import PIL
from ipywidgets import HTML, IntRangeSlider, FloatRangeSlider, jslink, Layout, VBox, HBox, Tab, Label, IntText, Dropdown, Text, Accordion, Button, Output, Textarea, FloatSlider, FloatText, Checkbox, SelectionSlider, Valid

global_keys = ['global', '', -1, '-1','global_settings']
no_preprocess_cn = ['control_sd21_qr','control_sd15_qr','control_sd15_temporalnet','control_sdxl_temporalnet_v1',
                            'control_sd15_ip2p','control_sd15_shuffle','control_sd15_inpaint','control_sd15_tile']

no_resolution_cn = ['control_sd21_qr','control_sd15_qr','control_sd15_temporalnet','control_sdxl_temporalnet_v1',
                            'control_sd15_ip2p','control_sd15_shuffle','control_sd15_inpaint','control_sd15_tile']
possible_controlnets = ['control_sd15_depth',
        'control_sd15_canny',
        'control_sd15_softedge',
        'control_sd15_mlsd',
        'control_sd15_normalbae',
        'control_sd15_openpose',
        'control_sd15_scribble',
        'control_sd15_seg',
        'control_sd15_temporalnet',
        'control_sd15_face',
        'control_sd15_ip2p',
        'control_sd15_inpaint',
        'control_sd15_lineart',
        'control_sd15_lineart_anime',
        'control_sd15_shuffle',
        'control_sd15_tile',
        'control_sd15_qr',
        'control_sd15_inpaint_softedge',
        'control_sd15_temporal_depth',
                            ]
possible_controlnets_sdxl = [
        'control_sdxl_canny',
        'control_sdxl_depth',
        'control_sdxl_softedge',
        'control_sdxl_seg',
        'control_sdxl_openpose',
        'control_sdxl_lora_128_depth',
        "control_sdxl_lora_256_depth",
        "control_sdxl_lora_128_canny",
        "control_sdxl_lora_256_canny",
        "control_sdxl_lora_128_softedge",
        "control_sdxl_lora_256_softedge",
        "control_sdxl_temporalnet_v1"
        ]
possible_controlnets_v2 = [
        'control_sd21_qr',
        "control_sd21_depth",
        "control_sd21_scribble",
        "control_sd21_openpose",
        "control_sd21_normalbae",
        "control_sd21_lineart",
        "control_sd21_softedge",
        "control_sd21_canny",
        "control_sd21_seg"
    ]
#adiff accepts all controlnets, just not prev stylized frame and cc masks, need to test temporal, inpaint
possible_controlnets_adiff = ['control_sd15_depth',
        'control_sd15_canny',
        'control_sd15_softedge',
        'control_sd15_mlsd',
        'control_sd15_normalbae',
        'control_sd15_openpose',
        'control_sd15_scribble',
        'control_sd15_seg',
        'control_sd15_face',
        'control_sd15_ip2p',
        'control_sd15_lineart',
        'control_sd15_lineart_anime',
        'control_sd15_shuffle',
        'control_sd15_tile',
        'control_sd15_qr'
    ]


def get_value(key, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            return obj[key].value
        else:
            for o in obj.keys():
                res = get_value(key, obj[o])
                if res is not None: return res
    if isinstance(obj, list):
        for o in obj:
            res = get_value(key, o)
            if res is not None: return res
    return None

def set_value(key, value, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
            obj[key].value = value
        else:
            for o in obj.keys():
                set_value(key, value, obj[o])

    if isinstance(obj, list):
        for o in obj:
            set_value(key, value, o)


class FilePath(HBox):
    def __init__(self,  **kwargs):
        self.model_path = Text(value='',  continuous_update = True,**kwargs)
        self.path_checker = Valid(
        value=False, layout=Layout(width='2000px')
        )

        self.model_path.observe(self.on_change)
        super().__init__([self.model_path, self.path_checker])

    def __getattr__(self, attr):
        if attr == 'value':
            return self.model_path.value
        else:
            return super.__getattr__(attr)

    def on_change(self, change):
        if change['name'] == 'value':
            if os.path.exists(change['new']):
                self.path_checker.value = True
                self.path_checker.description = ''
            else:
                self.path_checker.value = False
                self.path_checker.description = 'The file does not exist. Please specify the correct path.'

def add_labels_dict(gui):
    style = {'description_width': '250px' }
    layout = Layout(width='500px')
    gui_labels = {}
    for key in gui.keys():
        gui[key].style = style
        if isinstance(gui[key], ControlGUI):
          continue
        if not isinstance(gui[key], Textarea) and not isinstance( gui[key],Checkbox ): 
            gui[key].layout.width = '500px'
        if isinstance( gui[key],Checkbox ):
            html_label = HTML(
                description=gui[key].description,
                description_tooltip=gui[key].description_tooltip,  style={'description_width': 'initial' },
                layout = Layout(position='relative', left='-25px'))
            gui_labels[key] = HBox([gui[key],html_label])
            gui_labels[key].layout.visibility = gui[key].layout.visibility
            gui[key].description = ''
        else:
            gui_labels[key] = gui[key]
    return gui_labels

def set_globals_from_gui(user_settings_keys, guis, user_settings_eval_keys):
  for key in user_settings_keys:
    if key not in globals().keys():
      print(f'Variable {key} is not defined or present in globals()')
      continue
    #load mask clip

    if key in ['mask_clip_low', 'mask_clip_high']:
      value = get_value('mask_clip', guis)
    else:
      value = get_value(key, guis)

    if key in ['latent_fixed_mean', 'latent_fixed_std']:
      value = str(value)

    #apply eval for string schedules
    if key in user_settings_eval_keys:
      value = eval(value)

    if key == 'mask_clip_low':
      value = value[0]
    if key == 'mask_clip_high':
      value = value[1]

    globals()[key] = value

class ControlNetControls(HBox):
    def __init__(self,  name, values, **kwargs):
        self.label  = HTML(
                description=name,
                description_tooltip=name,  style={'description_width': 'initial' },
                layout = Layout(position='relative', left='-25px', width='200px'))
        self.name = name
        self.enable = Checkbox(value=values['weight']>0,description='',indent=True, description_tooltip='Enable model.',
                               style={'description_width': '25px' },layout=Layout(width='70px', left='-25px'))
        self.weight = FloatText(value = values['weight'], description=' ', step=0.05,
                                description_tooltip = 'Controlnet model weights. ',
                                layout=Layout(width='100px', visibility= 'visible' if values['weight']>0 else 'hidden'),
                                style={'description_width': '25px' })
        self.start_end = FloatRangeSlider(
          value=[values['start'],values['end']],
          min=0,
          max=1,
          step=0.01,
          description=' ',
          description_tooltip='Controlnet active step range settings. For example, [||||||||||] 50 steps,  [-------|||] 0.3 style strength (effective steps - 0.3x50 = 15), [--||||||--] - controlnet working range with start = 0.2 and end = 0.8, effective steps from 0.2x50 = 10 to 0.8x50 = 40',
          disabled=False,
          continuous_update=False,
          orientation='horizontal',
          readout=True,
          layout = Layout(width='300px', visibility= 'visible' if values['weight']>0 else 'hidden'),
          style={'description_width': '50px' }
        )


        if (not "preprocess" in values.keys()) or values["preprocess"] in global_keys:
          values["preprocess"] = 'global'

        if (not "mode" in values.keys()) or values["mode"] in global_keys:
          values["mode"] = 'global'

        if (not "detect_resolution" in values.keys()) or values["detect_resolution"] in global_keys:
          values["detect_resolution"] = -1


        if (not "source" in values.keys()) or values["source"] in global_keys:
          if name == 'control_sd15_inpaint': values["source"] = 'stylized'
          else: values["source"] = 'global'
        if values["source"] == 'init': values["source"] = 'raw_frame'


        self.preprocess = Dropdown(description='',
                           options = ['True', 'False', 'global'], value = values['preprocess'],
                           description_tooltip='Preprocess input for this controlnet', layout=Layout(width='80px'))

        self.mode = Dropdown(description='',
                           options = ['balanced', 'controlnet', 'prompt', 'global'], value = values['mode'],
                           description_tooltip='Controlnet mode. Pay more attention to controlnet prediction, to prompt or somewhere in-between.',
                             layout=Layout(width='100px'))

        self.detect_resolution = IntText(value = values['detect_resolution'], description='',
                                         description_tooltip = 'Controlnet detect_resolution.',layout=Layout(width='80px'), style={'description_width': 'initial' })

        self.source = Text(value=values['source'], description = '', layout=Layout(width='200px'),
                           description_tooltip='controlnet input source, either a file or video, raw_frame, cond_video, color_video, or stylized - to use previously stylized frame ad input. leave empty for global source')

        self.enable.observe(self.on_change)
        self.weight.observe(self.on_change)
        settings = [self.enable, self.label, self.weight, self.start_end, self.mode, self.source, self.detect_resolution, self.preprocess]
        # no_preprocess_cn = ['control_sd21_qr','control_sd15_qr','control_sd15_temporalnet','control_sdxl_temporalnet_v1',
        #                     'control_sd15_ip2p','control_sd15_shuffle','control_sd15_inpaint','control_sd15_tile']
        if name in no_preprocess_cn: self.preprocess.layout.visibility = 'hidden'
        # no_resolution_cn = ['control_sd21_qr','control_sd15_qr','control_sd15_temporalnet','control_sdxl_temporalnet_v1',
        #                     'control_sd15_ip2p','control_sd15_shuffle','control_sd15_inpaint','control_sd15_tile']
        if name in no_resolution_cn: self.detect_resolution.layout.visibility = 'hidden'

        if values['weight']==0:
              self.preprocess.layout.visibility = 'hidden'
              self.mode.layout.visibility = 'hidden'
              self.detect_resolution.layout.visibility = 'hidden'
              self.source.layout.visibility = 'hidden'
        super().__init__(settings, layout = Layout(valign='center'))

    def on_change(self, change):
      if change['name'] == 'value':
        if self.enable.value:
              self.weight.layout.visibility = 'visible'
              if change['old'] == False and self.weight.value==0:
                self.weight.value = 1
              self.start_end.layout.visibility = 'visible'
              self.preprocess.layout.visibility = 'visible'
              self.mode.layout.visibility = 'visible'
              self.detect_resolution.layout.visibility = 'visible'
              self.source.layout.visibility = 'visible'
        else:
              self.weight.layout.visibility = 'hidden'
              self.start_end.layout.visibility = 'hidden'
              self.preprocess.layout.visibility = 'hidden'
              self.mode.layout.visibility = 'hidden'
              self.detect_resolution.layout.visibility = 'hidden'
              self.source.layout.visibility = 'hidden'

    def __setattr__(self, attr, values):
        if attr == 'value':
          self.enable.value = values['weight']>0
          self.weight.value = values['weight']
          self.start_end.value=[values['start'],values['end']]
          if (not "preprocess" in values.keys()) or values["preprocess"] in global_keys:
                    values["preprocess"] = 'global'

          if (not "mode" in values.keys()) or values["mode"] in global_keys:
                    values["mode"] = 'global'

          if (not "detect_resolution" in values.keys()) or values["detect_resolution"] in global_keys:
                    values["detect_resolution"] = -1

          if (not "source" in values.keys()) or values["source"] in global_keys:
                    if self.name == 'control_sd15_inpaint': values["source"] = 'stylized'
                    else: values["source"] = 'global'
          if values["source"] == 'init': values["source"] = 'raw_frame'
          self.preprocess.value = values['preprocess']
          self.mode.value = values['mode']
          self.detect_resolution.value = values['detect_resolution']
          self.source.value=values['source']

        else: super().__setattr__(attr, values)

    def __getattr__(self, attr):
        if attr == 'value':
            weight = 0
            if self.weight.value>0 and self.enable.value: weight = self.weight.value
            (start,end) = self.start_end.value
            values = {
                  "weight": weight,
                  "start":start,
                  "end":end,

                }
            values['preprocess'] = self.preprocess.value
            values['mode'] = self.mode.value
            values['detect_resolution'] = self.detect_resolution.value
            values['source'] = self.source.value
            return values
        if attr == 'name':
          return self.name
        else:
            return super.__getattr__(attr)

class ControlGUI(VBox):
  def __init__(self, args, model_version='controlnet_multimodel'):
    enable_label = HTML(
                    description='Enable',
                    description_tooltip='Enable',  style={'description_width': '50px' },
                    layout = Layout(width='40px', left='-50px', ))
    model_label = HTML(
                    description='Model name',
                    description_tooltip='Model name',  style={'description_width': '100px' },
                    layout = Layout(width='265px'))
    weight_label = HTML(
                    description='weight',
                    description_tooltip='Model weight. 0 weight effectively disables the model. The total sum of all the weights will be normalized to 1.',  style={'description_width': 'initial' },
                    layout = Layout(position='relative', left='-25px', width='125px'))#65
    range_label = HTML(
                    description='active range (% or total steps)',
                    description_tooltip='Model`s active range. % of total steps when the model is active.\n Controlnet active step range settings. For example, [||||||||||] 50 steps,  [-------|||] 0.3 style strength (effective steps - 0.3x50 = 15), [--||||||--] - controlnet working range with start = 0.2 and end = 0.8, effective steps from 0.2x50 = 10 to 0.8x50 = 40',  style={'description_width': 'initial' },
                    layout = Layout(position='relative', left='-25px', width='200px'))
    mode_label = HTML(
                    description='mode',
                    description_tooltip='Controlnet mode. Pay more attention to controlnet prediction, to prompt or somewhere in-between.', layout = Layout(width='110px', left='0px', ))
    source_label = HTML(
                    description='source',
                    description_tooltip='controlnet input source, either a file or video, raw_frame, cond_video, color_video, or stylized - to use previously stylized frame ad input. leave empty for global source',
                    layout = Layout(width='210px', left='0px', ))
    resolution_label = HTML(
                    description='resolution',
                    description_tooltip='Controlnet detect_resolution. The size of the image fed into annotator model if current controlnet has one.',
                    layout = Layout(width='90px', left='0px', ))
    preprocess_label = HTML(
                    description='preprocess',
                    description_tooltip='Preprocess (put through annotator model) input for this controlnet. When disabled, puts raw image from selected source into the controlnet. For example, if you have sequence of pdeth maps from your 3d software, you need to put path to those maps into source field and disable preprocessing.',
                    layout = Layout(width='80px', left='0px', ))
    controls_list = [HBox([enable_label,model_label, weight_label, range_label, mode_label, source_label, resolution_label, preprocess_label ])]
    controls_dict = {}
    self.possible_controlnets = possible_controlnets
    if model_version == 'control_multi':
      self.possible_controlnets = possible_controlnets
    elif model_version == 'control_multi_sdxl':
      self.possible_controlnets = possible_controlnets_sdxl
    elif model_version in ['control_multi_v2','control_multi_v2_768']:
      self.possible_controlnets = possible_controlnets_v2
    elif model_version == 'control_multi_animatediff':
      self.possible_controlnets = possible_controlnets_adiff

    for key in self.possible_controlnets:
      if key in args.keys():
        w = ControlNetControls(key, args[key])
      else:
        w = ControlNetControls(key, {
            "weight":0,
            "start":0,
            "end":1
        })
        w.name = key
      controls_list.append(w)
      controls_dict[key] = w

    self.args = args
    self.ws = controls_dict
    super(ControlGUI, self).__init__(controls_list)

  def __setattr__(self, attr, values):
        if attr == 'value':
          keys = values.keys()
          for i in range(len(self.children)):
            w = self.children[i]
            if isinstance(w, ControlNetControls) :
              w.enable.value = False
              for key in values.keys():
                if w.name == key:
                  self.children[i].value = values[key]
        else:
          super().__setattr__(attr, values)

  def __getattr__(self, attr):
        if attr == 'value':
            res = {}
            for key in self.possible_controlnets:
              if self.ws[key].value['weight'] > 0:
                res[key] = self.ws[key].value
            return res
        else:
            return super.__getattr__(attr)

def set_visibility(key, value, obj):
    if isinstance(obj, dict):
        if key in obj.keys():
          obj[key].layout.visibility = value

def get_settings_from_gui(user_settings_keys, guis, user_settings_eval_keys, user_settings):
  for key in user_settings_keys:
    if key in ['mask_clip_low', 'mask_clip_high']:
      value = get_value('mask_clip', guis)
    else:
      value = get_value(key, guis)

    if key in ['latent_fixed_mean', 'latent_fixed_std']:
      value = str(value)

    #apply eval for string schedules
    if key in user_settings_eval_keys:
      try:
        value = eval(value)
      except Exception as e:
        print(e, key, value)

    #load mask clip
    if key == 'mask_clip_low':
      value = value[0]
    if key == 'mask_clip_high':
      value = value[1]

    user_settings[key] = value
  return user_settings

def infer_settings_path(path, settings_out):
    default_settings_path = path
    if default_settings_path == '-1':
      settings_files = sorted(glob(os.path.join(settings_out, '*.txt')),
                              key=os.path.getctime)
      if len(settings_files)>0:
        default_settings_path = settings_files[-1]
      else:
        print('Skipping load latest run settings: no settings files found.')
        return ''
    else:
      try:
        if type(eval(default_settings_path)) == int:
          files = sorted(glob(os.path.join(settings_out, '*.txt')))
          for f in files:
            if f'({default_settings_path})' in f:
              default_settings_path = f
      except: pass

    path = default_settings_path
    return path

def load_settings(path, guis):
    path = infer_settings_path(path)

    # global guis, load_settings_path, output
    global output
    if not os.path.exists(path):
        output.clear_output()
        print('Please specify a valid path to a settings file.')
        return guis
    if path.endswith('png'):
      img = PIL.Image.open(path)
      exif_data = img._getexif()
      settings = json.loads(exif_data[37510])

    else:
      print('Loading settings from: ', path)
      with open(path, 'rb') as f:
          settings = json.load(f)

    for key in settings:
        try:
            val = settings[key]
            if key == 'normalize_latent' and val == 'first_latent':
              val = 'init_frame'
              settings['normalize_latent_offset'] = 0
            if key == 'turbo_frame_skips_steps' and val == None:
                val = '100% (don`t diffuse turbo frames, fastest)'
            if key == 'seed':
                key = 'set_seed'
            if key == 'grad_denoised ':
                key = 'grad_denoised'
            if type(val) in [dict,list]:
                if type(val) in [dict]:
                  temp = {}
                  for k in val.keys():
                    temp[int(k)] = val[k]
                  val = temp
                val = json.dumps(val)
            if key == 'cc_masked_diffusion':
              key = 'cc_masked_diffusion_schedule'
              val = f'[{val}]'
            if key == 'mask_clip':
              val = eval(val)
            if key == 'sampler':
              val = getattr(K.sampling, val)
            if key == 'controlnet_multimodel':
              val = val.replace('control_sd15_hed', 'control_sd15_softedge')
              val = json.loads(val)
              set_value(key, val, guis)
              set_value(key, val, guis)
            set_value(key, val, guis)
        except Exception as e:
            print(key), print(settings[key] )
            print(e)
    print('Successfully loaded settings from ', path )
    return guis
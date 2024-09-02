---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.10/dist-packages/ipywidgets/widgets/interaction.py in update(self, *args)
    255                     value = widget.get_interact_value()
    256                     self.kwargs[widget._kwarg] = value
--> 257                 self.result = self.f(**self.kwargs)
    258                 show_inline_matplotlib_plots()
    259                 if self.auto_display and self.result is not None:

2 frames
<ipython-input-34-03fdf9b4247b> in find_max_inscribed_circle(upper_edge, lower_edge, mask)
     50             left_edge = max(x - width, 0)
     51             right_edge = min(x + width, width - 1)
---> 52             dist_to_left = min(np.abs(upper_edge[left_edge:x] - y), np.abs(lower_edge[left_edge:x] - y)).min() if x > 0 else width
     53             dist_to_right = min(np.abs(upper_edge[x:right_edge] - y), np.abs(lower_edge[x:right_edge] - y)).min() if x < width - 1 else width
     54 

ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

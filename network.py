import torch

def testSlomo(components, data, size):
    # Extract items
    (keyframe_left, keyframe_right), low_res_frames = data
    flowComputation, flowEnhancement, appearanceEstimation, resnetConv1, flowBackWarp = components

    flow_dict = {}
    output_frames = []

    for frame_index in range(len(low_res_frames) - 1):

        low_res_target = low_res_frames[frame_index]

        # Compute initial flow estimates using low resolution auxiliary frames
        flow_left_init, flow_right_init, flow_dict = \
            compute_flow(flowComputation, flowBackWarp, low_res_frames, frame_index, size, flow_dict)

        # Get warped keyframes using initial flow estimates
        warped_left_init  = flowBackWarp(keyframe_left,  flow_left_init)
        warped_right_init = flowBackWarp(keyframe_right, flow_right_init)

        # Generate enhanced flow and visibility maps
        output = flowEnhancement(torch.cat((keyframe_left, 
                                            keyframe_right, 
                                            low_res_target, 
                                            flow_right_init, 
                                            flow_left_init,
                                            warped_right_init, 
                                            warped_left_init), dim=1))

        flow_left  = output[:, :2, :, :] + flow_left_init
        flow_right = output[:, 2:4, :, :] + flow_right_init
        visibility_map_left  = torch.sigmoid(output[:, 4:5, :, :])
        visibility_map_right = 1 - visibility_map_left

        # Get warped keyframes using enhanced flows
        warped_left  = flowBackWarp(keyframe_left,  flow_left)
        warped_right = flowBackWarp(keyframe_right, flow_right)

        # Extract contextual information
        contextual_info_left   = resnetConv1(keyframe_left)
        contextual_info_right  = resnetConv1(keyframe_right)
        contextual_info_target = resnetConv1(low_res_target)

        # Warp contextual information
        warped_contextual_info_left  = flowBackWarp(contextual_info_left, flow_left)
        warped_contextual_info_right = flowBackWarp(contextual_info_right, flow_right)

        # Synthesize final intermediate frames (target)
        synthesized_intermediate_frame = \
            appearanceEstimation(torch.cat((visibility_map_left * warped_contextual_info_left, 
                                            visibility_map_right * warped_contextual_info_right, 
                                            contextual_info_target, 
                                            visibility_map_left * warped_left, 
                                            visibility_map_right * warped_right, 
                                            low_res_target), dim=1))

        synthesized_intermediate_frame = torch.clamp(synthesized_intermediate_frame, 0, 1)
        output_frames.append(synthesized_intermediate_frame[0].cpu().detach())

    return output_frames

def compute_flow(flowComputation, flowBackWarp, images, frame_index, size, flow_dict):
    flow_left_chain  = torch.zeros(((1,2,size[0],size[1]))).cuda()
    flow_right_chain = torch.zeros(((1,2,size[0],size[1]))).cuda()
    
    N = len(images)

    for idx in range(frame_index - 1, -1, -1):
        if "{}->{}".format(idx + 1, idx) in flow_dict:
            flow_left = flow_dict["{}->{}".format(idx + 1, idx)]
        else:
            flow_left  = 20.0 * torch.nn.functional.interpolate(input=flowComputation(torch.cat((images[idx + 1], images[idx]), axis=1)), size=size, mode='bilinear', align_corners=False)
            flow_dict["{}->{}".format(idx + 1, idx)] = flow_left
            
        flow_left_chain += flowBackWarp(flow_left, flow_left_chain)

    for idx in range(frame_index, N - 1):
        if "{}->{}".format(idx, idx + 1) in flow_dict:
            flow_right = flow_dict["{}->{}".format(idx, idx + 1)]
        else:
            flow_right  = 20.0 * torch.nn.functional.interpolate(input=flowComputation(torch.cat((images[idx], images[idx + 1]), axis=1)), size=size, mode='bilinear', align_corners=False)
            flow_dict["{}->{}".format(idx, idx + 1)] = flow_right
            
        flow_right_chain += flowBackWarp(flow_right, flow_right_chain)
    
    return flow_left_chain, flow_right_chain, flow_dict

def getPadding(intWidth, intHeight):
    if intWidth != ((intWidth >> 6) << 6):
        intWidth_pad = (((intWidth >> 6) + 1) << 6)  # more than necessary
        intPaddingLeft =int((intWidth_pad - intWidth) / 2)
        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    else:
        intWidth_pad = intWidth
        intPaddingLeft = 0
        intPaddingRight= 0

    if intHeight != ((intHeight >> 6) << 6):
        intHeight_pad = (((intHeight >> 6) + 1) << 6)  # more than necessary
        intPaddingTop = int((intHeight_pad - intHeight) / 2)
        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    else:
        intHeight_pad = intHeight
        intPaddingTop = 0
        intPaddingBottom = 0

    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

    return pader, intWidth_pad, intHeight_pad
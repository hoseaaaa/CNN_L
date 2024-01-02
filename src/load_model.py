def load_model(model, filepath):
    # Load the saved model
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode

import slack


class NotifySlack:
    def __init__(self, token, channel):
        self.channel = '#%s' % channel
        self.client = slack.WebClient(token=token)

    def notify(self, message):
        self.client.chat_postMessage(channel=self.channel, text=message)

import { handleConversations } from '../scripts/server.mjs';

export default function handler(req, res) {
  return handleConversations(req, res);
}
